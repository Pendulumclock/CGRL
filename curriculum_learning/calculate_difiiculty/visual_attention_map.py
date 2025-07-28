
import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from PIL import Image, ImageDraw
import torch
import numpy as np
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
import cv2
import argparse

def extract_last_layer_cross_attention(model, inputs, img, target_start_index, target_end_index, first_image_pad_index):
    """
    Extract the cross-attention weights from each decoder layer of the model
    given the inputs (image + text).

    The function returns the cross-attention averaged over all text tokens 
    within the specified range, resulting in a tensor of shape (batch_size, src_len).

    Args:
        model: The AutoModelForVision2Seq with output_attentions=True.
        inputs: A dictionary returned by processor(prompt, images, return_tensors="pt")
                and moved to device via .to(device).
        img: The original PIL image, used to compute patch grid dimensions.
        target_start_index: Start index of target text tokens in decoder input.
        target_end_index: End index (exclusive) of target text tokens.
        first_image_pad_index: The index of the first image token in encoder input.

    Returns:
        avg_attention: A numpy array of shape (batch_size, src_len)
                       representing the cross-attention averaged over specified tokens.
        patch_size_info: A tuple (num_patches, grid_h, grid_w)
                         with patch grid dimensions for later reshaping.
    """
    # 1) Ensure attentions are returned.
    #    Hugging Face models accept output_attentions=True to include attention tensors in outputs.
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,   # Required to return attentions
            return_dict=True,
            use_cache=False  
        )

    # 2) decoder_cross_attentions is a tuple with cross-attention tensors from each decoder layer.
    #    Typical shape: len = number of decoder layers,
    #    each element: (batch_size, num_heads, tgt_len, src_len)
    decoder_cross_attentions = outputs.attentions

    # 3) Get patch-related info from the vision encoder config.
    #    For example, ViT-B_16: patch_size=16, image_size=224 → 224/16 = 14 patches per side → 14x14 grid = 196 patches.
    vision_config = model.visual.config

    # 4) Compute num_patches, grid_h, grid_w.
    # img_size = vision_config.image_size  # e.g., 224
    w, h = img.size  # Get original image width and height
    patch_size = vision_config.patch_size * 2  # e.g., 14 or 16 (doubled if needed)
    grid_h = round(h / patch_size)
    grid_w = round(w / patch_size)
    num_patches = grid_h * grid_w  # e.g., 14*14 = 196

    # Compute weighted cross-attention.
    attn_mean_text_weight = 0.0
    num_layers = len(decoder_cross_attentions)
    weights = torch.linspace(1, num_layers, steps=num_layers, device=decoder_cross_attentions[0].device)  # Weights: 1 to N
    weights = weights / weights.sum()

    for i, attn in enumerate(decoder_cross_attentions):
        # [B, H, T, S] → [B, H, S_image]: select cross-attention between target text tokens and image tokens.
        attn_slice = attn[:, :, target_start_index:target_end_index, first_image_pad_index:num_patches+first_image_pad_index].mean(dim=2)
        attn_mean = attn_slice.mean(dim=1)  # [B, S_image]: average over heads
        attn_mean_text_weight += weights[i] * attn_mean  # Weighted sum

    src_len = num_patches
    # Return the final averaged attention and patch grid info.
    return attn_mean_text_weight.detach().cpu().numpy(), (src_len, grid_h, grid_w)



def visualize_attention_on_image(
    orig_image_pil,        # Original RGB image in PIL.Image format
    attn_weights,          # Numpy array: (src_len,) or (batch, src_len)
    patch_info,            # Tuple: (src_len, grid_h, grid_w), see output of previous step
    alpha: float = 0.5     # Overlay transparency, range 0.0~1.0
):
    # 1) If batched, assume batch_size=1 and take the first sample
    if attn_weights.ndim == 2:
        attn_weights = attn_weights[0]  # (src_len,)

    src_len, grid_h, grid_w = patch_info
    orig_w, orig_h = orig_image_pil.size  # PIL.Image.size returns (width, height)

    # 2) Convert weights to float32 for numeric stability
    weights = attn_weights.astype(np.float32)  # (src_len,)

    # 4) Reshape to (grid_h, grid_w)
    heatmap = weights.reshape(grid_h, grid_w)  # (grid_h, grid_w)

    # 5) Normalize to [0, 1]
    heatmap -= heatmap.min()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # 6) Resize heatmap to original image size using bilinear interpolation
    heatmap_resized = cv2.resize(
        heatmap,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )  # (orig_h, orig_w)

    # 7) Convert heatmap to uint8 [0, 255] and apply COLORMAP_JET
    heatmap_uint8 = np.uint8(255 * heatmap_resized)        # (H, W)
    color_mask = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR (H, W, 3)

    # 8) Convert original PIL RGB image to OpenCV BGR format
    orig_rgb = np.array(orig_image_pil.resize((orig_w, orig_h)))  # (H, W, 3), RGB
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)          # To BGR

    # 9) Overlay heatmap on original image
    # src1: background (original), src2: foreground (heatmap), alpha controls transparency
    overlayed = cv2.addWeighted(src1=orig_bgr, alpha=1 - alpha,
                                src2=color_mask, beta=alpha, gamma=0.0)

    # 10) If displaying with matplotlib, convert BGR back to RGB
    overlayed_rgb = cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)  # (H, W, 3)

    return overlayed_rgb, heatmap_resized

# Get target region mask
def get_target_area(data):
    image_size = data['image_size']
    surrounding_coordinates = data['surrounding_coordinates']

    # Convert to tuple format
    polygon_points = [tuple(pt) for pt in surrounding_coordinates]

    # Create a black mask image (0 means outside)
    mask_img = Image.new("L", image_size, 0)  # "L" means single-channel grayscale

    # Draw the polygon region, fill value = 1 (or 255, then normalize later)
    ImageDraw.Draw(mask_img).polygon(polygon_points, outline=1, fill=1)

    # Convert to NumPy array
    mask_array = np.array(mask_img)

    return mask_array

# Display the original image with the target area and the overlaid heatmap
def show_location(data, overlayed, save_path):
    surrounding_coordinates = data['surrounding_coordinates']

    # Convert to PIL Image object
    ori_image = Image.fromarray(overlayed)

    # Convert to tuple format
    polygon_points = [tuple(pt) for pt in surrounding_coordinates]

    draw = ImageDraw.Draw(ori_image)

    # Draw the closed polygon outline
    draw.polygon(polygon_points, outline="red", fill=None)

    # Draw thick polygon edges
    for i in range(len(polygon_points)):
        p1 = polygon_points[i]
        p2 = polygon_points[(i + 1) % len(polygon_points)]
        draw.line([p1, p2], fill=(255, 0, 0), width=8)

    plt.figure(figsize=(8, 8))
    plt.imshow(ori_image)
    plt.title("Heatmap Overlay with Target Area")
    plt.axis("off")
    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def get_attention_map(model, processor, tokenizer, device, navGym):
    image_path = navGym['image_path']  # navGym.cur_whole_map
    orig_img = Image.open(image_path).convert("RGB")
    # Downscale the original image to reduce GPU memory usage
    width, height = orig_img.size
    resized_img = orig_img.resize((width // 2, height // 2))

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": resized_img,
                },
                {"type": "text", "text": navGym['target_description']},
            ],
        }
    ]

    ### The text description we want to visualize attention for
    target_text = navGym['target_description']
    target_input_ids = tokenizer(target_text)["input_ids"]

    # Format the message using the chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Extract vision content (images or videos) from messages
    image_inputs, video_inputs = process_vision_info(messages)

    # Convert text and vision inputs to model input format
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,             # Automatically pad sequences to equal length
        return_tensors="pt",      # Return PyTorch tensors
    )

    # target_text is the description, target_input_ids is its tokenized form
    n = len(target_input_ids)

    # Find the index of the first image pad token
    first_image_pad_index = inputs['input_ids'][0].tolist().index(
        tokenizer.convert_tokens_to_ids("<|vision_start|>")
    ) + 1

    # Locate the start and end index of the target text segment in the input sequence
    for i in range(len(inputs['input_ids'][0]) - n + 1):
        if inputs['input_ids'][0][i:i+n].tolist() == target_input_ids:
            target_start_index = i
            target_end_index = i + n
            break

    inputs = inputs.to(device)

    attn_weights, patch_info = extract_last_layer_cross_attention(
        model,
        inputs,
        resized_img,
        target_start_index,
        target_end_index,
        first_image_pad_index
    )

    # Generate the overlay image and normalized heatmap for difficulty calculation
    overlayed, cal_heatmap = visualize_attention_on_image(
        orig_image_pil=orig_img,
        attn_weights=attn_weights,  # (batch_size, src_len)
        patch_info=patch_info,
        alpha=0.4
    )

    return overlayed, cal_heatmap

# IoU_soft
def cal_difficulty(attention_map,target_area):
    binary_map = attention_map
    cross = (binary_map * target_area).sum()
    sum_A = binary_map.sum()
    sum_M = target_area.sum()
    difficulty = 1 - (cross / (sum_M + sum_A - cross))
    return difficulty

def main():
    # Run this file under the CGRL directory
    parser = argparse.ArgumentParser()
    # Add model name argument
    parser.add_argument('--model_name', default='Qwen/Qwen2.5-VL-7B-Instruct', type=str)
    args = parser.parse_args()

    # ---- 0. Load data ---- #
    citynavDataPath = "curriculum_learning/data/data_example.json"
    with open(citynavDataPath, 'r') as f:
        citynavData = json.load(f)
    print(f"Loaded {len(citynavData)} samples from {citynavDataPath}")
    # Print the first sample to check data format
    print(f"Sample 0: {citynavData[0]}")  

    # ---- 1. Set device ---- #
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 2. Load model & processor ---- #
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,   # Use half precision to reduce GPU memory usage
        device_map="auto"            # Automatically split model across available GPUs
    )
    print("Model loaded")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model.eval()
    model.config.use_cache = False

    dataSize = len(citynavData)

    # ---- 3. Generate heatmaps & compute difficulty ---- #
    for i in range(dataSize):
        navGym = citynavData[i]
        with torch.no_grad():
            overlayed, attention_map = get_attention_map(model, processor, tokenizer, device, navGym)
        target_area = get_target_area(navGym)
        difficulty = cal_difficulty(attention_map, target_area)
        navGym["difficulty"] = difficulty
        label = "_".join(str(x) for x in navGym['episode_id'])
        show_location(navGym, overlayed, f'curriculum_learning/heatmap/{label}.jpg')
        print(f"Sample {label} processed, difficulty: {difficulty:.5f}, count: {i+1}/{dataSize}")
        # Clear CUDA cache
        del attention_map, overlayed
        torch.cuda.empty_cache()

    # ---- 4. Save results ---- #
    with open("curriculum_learning/data/data_example_difficulty.json", "w", encoding="utf-8") as f:
        json.dump(citynavData, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    main()
