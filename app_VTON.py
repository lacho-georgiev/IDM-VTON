import argparse
import os
import io
import base64
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL
from typing import List, Optional
from util.common import open_folder
from util.image import pil_to_binary_mask, save_output_image
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image
from util.pipeline import quantize_4bit, restart_cpu_offload, torch_gc
import logging
import time

app = FastAPI()

# Allow CORS for testing purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--share", type=str, default=False, help="Set to True to share the app publicly.")
parser.add_argument("--lowvram", action="store_true", help="Enable CPU offload for model operations.")
parser.add_argument("--load_mode", default=None, type=str, choices=["4bit", "8bit"],
                    help="Quantization mode for optimization memory consumption")
parser.add_argument("--fixed_vae", action="store_true", default=True, help="Use fixed vae for FP16.")
args = parser.parse_args()

load_mode = args.load_mode
fixed_vae = args.fixed_vae

dtype = torch.float16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = 'yisol/IDM-VTON'
vae_model_id = 'madebyollin/sdxl-vae-fp16-fix'

dtypeQuantize = dtype

if load_mode in ('4bit', '8bit'):
    dtypeQuantize = torch.float8_e4m3fn

ENABLE_CPU_OFFLOAD = args.lowvram
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.allow_tf32 = False
need_restart_cpu_offloading = False

unet = None
pipe = None
UNet_Encoder = None
example_path = os.path.join(os.path.dirname(__file__), 'example')


@app.on_event("startup")
async def load_models():
    global unet, pipe, UNet_Encoder
    logger.info("Initializing models")
    model_init_start = time.time()

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
        vae = AutoencoderKL.from_pretrained(
            model_id,
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

    logger.info(f"Model initialization completed in {time.time() - model_init_start:.2f}s")


def image_to_base64(img: Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def start_tryon(dict, garm_img, garment_des, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed,
                seed, number_of_images, prompt):  # Add prompt parameter
    global pipe, unet, UNet_Encoder, need_restart_cpu_offloading

    logger.info("Starting tryon process")
    start_time = time.time()

    torch_gc()
    parsing_model = Parsing(0)
    openpose_model = OpenPose(0)
    openpose_model.preprocessor.body_estimation.model.to(device)
    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    if need_restart_cpu_offloading:
        restart_cpu_offload(pipe, load_mode)
    elif ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()

    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img_orig = dict["background"].convert("RGB")

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

    if is_checked:
        try:
            mask_start = time.time()
            keypoints = openpose_model(human_img.resize((384, 512)))
            model_parse, _ = parsing_model(human_img.resize((384, 512)))
            mask, mask_gray = get_mask_location('hd', category, model_parse, keypoints)
            mask = mask.resize((768, 1024))
            logger.info(f"Mask generation completed in {time.time() - mask_start:.2f}s")
        except Exception as e:
            logger.error(f"Error during mask generation: {e}")
            return {"error": "Could not find a human in the photo."}, None
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm',
        '-v', '--opts', 'MODEL.DEVICE', 'cuda')
    )
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    if pipe.text_encoder is not None:
        pipe.text_encoder.to(device)

    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.to(device)

    logger.info("Starting image generation")
    generation_start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            with torch.no_grad():
                # Use the prompt parameter for the prompt string
                if prompt:
                    input_prompt = prompt
                else:
                    input_prompt = "model is wearing " + garment_des

                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        input_prompt,
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

                    pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, dtype)
                    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, dtype)
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
                        logger.info(f"Image generation completed in {time.time() - generation_start:.2f}s")
                    return results, None
    logger.info(f"Total tryon process completed in {time.time() - start_time:.2f}s")


@app.post("/tryon")
async def tryon(
        background_tasks: BackgroundTasks,
        model_image: UploadFile = File(...),
        garment_image: UploadFile = File(...),
        description: str = Form(""),
        category: str = Form("upper_body"),  # Add category parameter
        prompt: str = Form(""),  # Add prompt parameter
        is_checked: bool = Form(True),
        is_checked_crop: bool = Form(True),
        denoise_steps: int = Form(30),
        is_randomize_seed: bool = Form(True),
        seed: int = Form(1),
        number_of_images: int = Form(1)
):
    request_start = time.time()
    try:
        logger.info("Received tryon request")
        model_img = Image.open(io.BytesIO(await model_image.read()))
        garment_img = Image.open(io.BytesIO(await garment_image.read()))

        dict = {"background": model_img, "layers": [model_img]}  # Simplified example; adjust as needed

        results, error = start_tryon(
            dict, garment_img, description, category, is_checked, is_checked_crop, denoise_steps, is_randomize_seed,
            seed, number_of_images, prompt  # Pass new parameters
        )

        if error:
            logger.error(f"Error in tryon process: {error}")
            return JSONResponse(status_code=400, content={"error": error})

        results_base64 = [image_to_base64(Image.open(img_path)) for img_path in results]

        logger.info(f"Request processed in {time.time() - request_start:.2f}s")
        return JSONResponse(content={"output_images": results_base64})
    except Exception as e:
        logger.error(f"Exception during tryon process: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
