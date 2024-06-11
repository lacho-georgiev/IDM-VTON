import os
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
import base64

from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc

app = FastAPI()

load_mode = os.getenv('LOAD_MODE', None)
fixed_vae = os.getenv('FIXED_VAE', True)

dtype = torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = 'yisol/IDM-VTON'
vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

dtypeQuantize = dtype

if load_mode in ('4bit', '8bit'):
    dtypeQuantize = torch.float8_e4m3fn

ENABLE_CPU_OFFLOAD = os.getenv('LOW_VRAM', False)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

unet = None
pipe = None
UNet_Encoder = None
example_path = os.path.join(os.path.dirname(__file__), 'example')


class TryOnRequest(BaseModel):
    dict: Optional[dict]
    garm_img: UploadFile
    garment_des: str
    category: str
    is_checked: bool
    is_checked_crop: bool
    denoise_steps: int
    is_randomize_seed: bool
    seed: int
    number_of_images: int


@app.on_event("startup")
async def startup_event():
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    if pipe is None:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=dtypeQuantize,
        )
        if load_mode == '4bit':
            quantize_4bit(unet)

        unet.requires_grad_(False)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float16,
        )
        if load_mode == '4bit':
            quantize_4bit(image_encoder)

        if fixed_vae:
            vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_pretrained(model_id,
                                                subfolder="vae",
                                                torch_dtype=dtype,
                                                )

        UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
            model_id,
            subfolder="unet_encoder",
            torch_dtype=dtypeQuantize,
        )

        if load_mode == '4bit':
            quantize_4bit(UNet_Encoder)

        UNet_Encoder.requires_grad_(False)
        image_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        unet.requires_grad_(False)

        pipe_param = {
            'pretrained_model_name_or_path': model_id,
            'unet': unet,
            'torch_dtype': dtype,
            'vae': vae,
            'image_encoder': image_encoder,
            'feature_extractor': CLIPImageProcessor(),
        }

        pipe = TryonPipeline.from_pretrained(**pipe_param).to(device)
        pipe.unet_encoder = UNet_Encoder
        pipe.unet_encoder.to(pipe.unet.device)

        if load_mode == '4bit':
            if pipe.text_encoder is not None:
                quantize_4bit(pipe.text_encoder)
            if pipe.text_encoder_2 is not None:
                quantize_4bit(pipe.text_encoder_2)
    else:
        if ENABLE_CPU_OFFLOAD:
            need_restart_cpu_offloading = True

    torch_gc()


@app.post("/tryon")
async def start_tryon(dict: Optional[str] = Form(None),
                      garm_img: UploadFile = File(...),
                      garment_des: str = Form(...),
                      category: str = Form(...),
                      is_checked: bool = Form(...),
                      is_checked_crop: bool = Form(...),
                      denoise_steps: int = Form(...),
                      is_randomize_seed: bool = Form(...),
                      seed: int = Form(...),
                      number_of_images: int = Form(...)):
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    garm_img_data = await garm_img.read()
    garm_img = Image.open(BytesIO(garm_img_data)).convert("RGB").resize((768, 1024))

    if dict:
        dict = eval(dict)
        human_img_orig = Image.open(BytesIO(base64.b64decode(dict["background"]))).convert("RGB")
    else:
        return JSONResponse(status_code=400, content={"error": "No input image provided."})

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    tensor_transfrom = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if need_restart_cpu_offloading:
        restart_cpu_offload(pipe, load_mode)
    elif ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()

    if is_checked:
        keypoints = openpose_model(human_img.resize((384, 512)))
        model_parse, _ = parsing_model(human_img.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(Image.open(BytesIO(base64.b64decode(dict['layers'][0]))).convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)

    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt = "a photo of " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(prompt, List):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device, dtype)
                garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device, dtype)
                results = []
                current_seed = seed
                for i in range(number_of_images):
                    if is_randomize_seed:
                        current_seed = torch.randint(0, 2 ** 32, size=(1,)).item()
                    generator = torch.Generator(device).manual_seed(current_seed) if seed != -1 else None
                    current_seed = current_seed + i

                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device, dtype),
                        negative_prompt_embeds=negative_prompt_embeds.to(device, dtype),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, dtype),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device, dtype),
                        text_embeds_cloth=prompt_embeds_c.to(device, dtype),
                        cloth=garm_tensor.to(device, dtype),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768, 1024)),
                        guidance_scale=2.0,
                        dtype=dtype,
                        device=device,
                    )[0]
                    if is_checked_crop:
                        out_img = images[0].resize(crop_size)
                        human_img_orig.paste(out_img, (int(left), int(top)))
                        img_path = save_output_image(human_img_orig, base_path="outputs", base_filename='img',
                                                     seed=current_seed)
                        results.append(img_path)
                    else:
                        img_path = save_output_image(images[0], base_path="outputs", base_filename='img')
                        results.append(img_path)

                result_images = [base64.b64encode(open(img, "rb").read()).decode() for img in results]
                return {"results": result_images, "mask_gray": base64.b64encode(mask_gray).decode()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
