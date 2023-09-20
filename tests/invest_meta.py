import os

# Define the new directory you want to change to
new_directory = '../.'

# Get the absolute path
abs_directory = os.path.abspath(new_directory)

# Check if the last segment of the path is 'synthtiger'
if os.path.basename(abs_directory) == 'synthtiger':
    # Change the current working directory
    os.chdir(new_directory)
    # Print the current working directory to verify the change
    print(os.getcwd())
elif os.path.basename(os.path.abspath("./")) == 'synthtiger':
    print("All fine no need to change")
else:
    print("The last segment of the path is not 'synthtiger'")
    raise Exception("Directory mismatch!")

import json
import os

import cv2
import numpy as np
from PIL import Image

from synthtiger import components, layers, templates, utils

BLEND_MODES = [
    "normal",
    "multiply",
    "screen",
    "overlay",
    "hard_light",
    "soft_light",
    "dodge",
    "divide",
    "addition",
    "difference",
    "darken_only",
    "lighten_only",
]


class Keys:
    POST = 'post'
    BLEND = 'blend'
    BACK_TEXTURE = 'back_texture'
    COLORMAP2 = 'colormap2'
    COLORMAP3 = 'colormap3'
    FG_STYLE = 'fg_style'
    QUALITY = 'quality'
    SHAPE = 'shape'
    FONT = 'font'
    CORPUS = "corpus"
    LAYOUT = 'layout'
    TEXTURE = 'texture'
    TRANSFORM = 'transform'
    FIT = 'fit'
    PAD = 'pad'


class SynthTiger(templates.Template):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.coord_output = config.get("coord_output", True)
        self.mask_output = config.get("mask_output", True)
        self.glyph_coord_output = config.get("glyph_coord_output", True)
        self.glyph_mask_output = config.get("glyph_mask_output", True)
        self.vertical = config.get("vertical", False)
        self.quality = config.get("quality", [95, 95])
        self.visibility_check = config.get("visibility_check", False)
        self.corpus = components.Selector(
            [
                components.BaseCorpus(),
            ],
            **config.get("corpus", {}),
        )
        self.font = components.BaseFont(**config.get("font", {}))
        self.texture = components.Switch(
            components.BaseTexture(), **config.get("texture", {})
        )
        self.colormap2 = components.GrayMap(**config.get("colormap2", {}))
        self.colormap3 = components.GrayMap(**config.get("colormap3", {}))
        self.color = components.Gray(**config.get("color", {}))
        self.shape = components.Switch(
            components.Selector(
                [components.ElasticDistortion(), components.ElasticDistortion()]
            ),
            **config.get("shape", {}),
        )
        self.layout = components.Selector(
            [components.FlowLayout(), components.CurveLayout()],
            **config.get("layout", {}),
        )
        self.style = components.Switch(
            components.Selector(
                [
                    components.TextBorder(),
                ]
            ),
            **config.get("style", {}),
        )
        self.transform = components.Switch(
            components.Selector(
                [
                    components.Perspective(),
                    components.Perspective(),
                    components.Trapezoidate(),
                    components.Trapezoidate(),
                    components.Skew(),
                    components.Skew(),
                    components.Rotate(),
                ]
            ),
            **config.get("transform", {}),
        )
        self.fit = components.Fit()
        self.pad = components.Switch(components.Pad(), **config.get("pad", {}))
        self.postprocess = components.Iterator(
            [
                components.Switch(components.AdditiveGaussianNoise()),
                components.Switch(components.GaussianBlur()),
                components.Switch(components.Resample()),
                components.Switch(components.MedianBlur()),
            ],
            **config.get("postprocess", {}),
        )

    def generate(self):
        return self.generate_from_meta(input_meta={})

    def generate_from_meta(self, input_meta):
        output_meta = {}
        if input_meta.get(Keys.QUALITY) is not None:
            quality = input_meta.get(Keys.QUALITY)
        else:
            quality = np.random.randint(self.quality[0], self.quality[1] + 1)
        output_meta[Keys.QUALITY] = quality

        fg_color, fg_style, bg_color = self._generate_color(input_meta, output_meta)
        fg_image, label, bboxes, glyph_fg_image, glyph_bboxes = self._generate_text(
            fg_color, fg_style, input_meta=input_meta, meta=output_meta
        )

        back_layer = layers.RectLayer(fg_image.shape[:2][::-1])
        self.color.apply([back_layer], bg_color)
        output_meta[Keys.BACK_TEXTURE] = self.texture.apply([back_layer], meta=input_meta.get(Keys.BACK_TEXTURE))
        bg_image = back_layer.output()

        image = _blend_images(fg_image, bg_image, self.visibility_check, input_meta, output_meta)

        image, fg_image, glyph_fg_image = self._postprocess_images(
            [image, fg_image, glyph_fg_image], input_meta, output_meta
        )

        return {
            "image": image,
            "label": label,
            "quality": quality,
            "mask": fg_image[..., 3],
            "bboxes": bboxes,
            "glyph_mask": glyph_fg_image[..., 3],
            "glyph_bboxes": glyph_bboxes,
            "meta": output_meta,
        }

    def init_save(self, root):
        os.makedirs(root, exist_ok=True)

        gt_path = os.path.join(root, "gt.txt")
        coords_path = os.path.join(root, "coords.txt")
        glyph_coords_path = os.path.join(root, "glyph_coords.txt")
        meta_path = os.path.join(root, "meta.jsonl")

        self.gt_file = open(gt_path, "w", encoding="utf-8")
        self.meta_file = open(meta_path, "w", encoding="utf-8")
        if self.coord_output:
            self.coords_file = open(coords_path, "w", encoding="utf-8")
        if self.glyph_coord_output:
            self.glyph_coords_file = open(glyph_coords_path, "w", encoding="utf-8")

    def save(self, root, data, idx):
        image = data["image"]
        label = data["label"]
        quality = data["quality"]
        mask = data["mask"]
        bboxes = data["bboxes"]
        glyph_mask = data["glyph_mask"]
        glyph_bboxes = data["glyph_bboxes"]

        image = Image.fromarray(image[..., :3].astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        glyph_mask = Image.fromarray(glyph_mask.astype(np.uint8))

        coords = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
        coords = "\t".join([",".join(map(str, map(int, coord))) for coord in coords])
        glyph_coords = [[x, y, x + w, y + h] for x, y, w, h in glyph_bboxes]
        glyph_coords = "\t".join(
            [",".join(map(str, map(int, coord))) for coord in glyph_coords]
        )

        shard = str(idx // 10000)
        image_key = os.path.join("images", shard, f"{idx}.jpg")
        mask_key = os.path.join("masks", shard, f"{idx}.png")
        glyph_mask_key = os.path.join("glyph_masks", shard, f"{idx}.png")
        image_path = os.path.join(root, image_key)
        mask_path = os.path.join(root, mask_key)
        glyph_mask_path = os.path.join(root, glyph_mask_key)

        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        image.save(image_path, quality=quality)
        if self.mask_output:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            mask.save(mask_path)
        if self.glyph_mask_output:
            os.makedirs(os.path.dirname(glyph_mask_path), exist_ok=True)
            glyph_mask.save(glyph_mask_path)

        self.gt_file.write(f"{image_key}\t{label}\n")
        meta = data["meta"]
        # Create a new dictionary with 'image_key' and 'label' first
        new_meta = {
            "final_image_path": image_key,
            "label": label
        }

        # Update the new dictionary with the original meta dictionary
        new_meta.update(meta)
        compressed_oneline_json = json.dumps(new_meta, ensure_ascii=False)
        self.meta_file.write(f"{compressed_oneline_json}\n")
        if self.coord_output:
            self.coords_file.write(f"{image_key}\t{coords}\n")
        if self.glyph_coord_output:
            self.glyph_coords_file.write(f"{image_key}\t{glyph_coords}\n")

    def end_save(self, root):
        self.gt_file.close()
        self.meta_file.close()
        if self.coord_output:
            self.coords_file.close()
        if self.glyph_coord_output:
            self.glyph_coords_file.close()

    def _generate_color(self, input_meta, output_meta):
        fg_style = self.style.sample(meta=input_meta.get(Keys.FG_STYLE))

        if fg_style["state"]:
            output_meta[Keys.COLORMAP3] = self.colormap3.sample(meta=input_meta.get(Keys.COLORMAP3))
            fg_color, bg_color, style_color = output_meta[Keys.COLORMAP3]
            fg_style["meta"]["meta"]["rgb"] = style_color["rgb"]
        else:
            output_meta[Keys.COLORMAP2] = self.colormap2.sample(meta=input_meta.get(Keys.COLORMAP2))
            fg_color, bg_color = output_meta[Keys.COLORMAP2]

        output_meta[Keys.FG_STYLE] = fg_style

        return fg_color, fg_style, bg_color

    def _generate_text(self, color, style, input_meta, meta):
        meta[Keys.CORPUS] = self.corpus.sample(meta=input_meta.get(Keys.CORPUS))
        label = self.corpus.data(meta[Keys.CORPUS])

        # for script using diacritic, ligature and RTL
        chars = utils.split_text(label, reorder=True)
        text = "".join(chars)

        default_font_meta = {"text": text, "vertical": self.vertical}
        if input_meta.get(Keys.FONT) is not None:
            default_font_meta.update(input_meta.get(Keys.FONT))
        font = self.font.sample(default_font_meta)
        meta[Keys.FONT] = font

        char_layers = [layers.TextLayer(char, **font) for char in chars]

        meta[Keys.SHAPE] = self.shape.apply(char_layers, meta=input_meta.get(Keys.SHAPE))
        default_layout_meta = {"meta": {"vertical": self.vertical}}
        if input_meta.get(Keys.LAYOUT) is not None:
            default_layout_meta = input_meta.get(Keys.LAYOUT)
        meta[Keys.LAYOUT] = self.layout.apply(char_layers, default_layout_meta)
        char_glyph_layers = [char_layer.copy() for char_layer in char_layers]

        text_layer = layers.Group(char_layers).merge()
        text_glyph_layer = text_layer.copy()

        self.color.apply([text_layer, text_glyph_layer], color)
        meta[Keys.TEXTURE] = self.texture.apply([text_layer, text_glyph_layer], meta=input_meta.get(Keys.TEXTURE))
        self.style.apply([text_layer, *char_layers], style)
        meta[Keys.TRANSFORM] = self.transform.apply(
            [text_layer, text_glyph_layer, *char_layers, *char_glyph_layers], meta=input_meta.get(Keys.TRANSFORM)
        )
        meta[Keys.FIT] = self.fit.apply([text_layer, text_glyph_layer, *char_layers, *char_glyph_layers],
                                        meta=input_meta.get(Keys.FIT))
        meta[Keys.PAD] = self.pad.apply([text_layer], meta=input_meta.get(Keys.PAD))

        for char_layer in char_layers:
            char_layer.topleft -= text_layer.topleft
        for char_glyph_layer in char_glyph_layers:
            char_glyph_layer.topleft -= text_layer.topleft

        out = text_layer.output()
        bboxes = [char_layer.bbox for char_layer in char_layers]

        glyph_out = text_glyph_layer.output(bbox=text_layer.bbox)
        glyph_bboxes = [char_glyph_layer.bbox for char_glyph_layer in char_glyph_layers]

        return out, label, bboxes, glyph_out, glyph_bboxes

    def _postprocess_images(self, images, input_meta, meta):
        image_layers = [layers.Layer(image) for image in images]
        meta[Keys.POST] = self.postprocess.apply(image_layers, input_meta.get(Keys.POST))
        return [image_layer.output() for image_layer in image_layers]


def _blend_images(src, dst, visibility_check, input_meta, meta):
    if input_meta.get(Keys.BLEND) is not None:
        blend_modes = input_meta.get(Keys.BLEND)
    else:
        blend_modes = np.random.permutation(BLEND_MODES)
        blend_modes = blend_modes.tolist()

    for blend_mode in blend_modes:
        out = utils.blend_image(src, dst, mode=blend_mode)
        if not visibility_check or _check_visibility(out, src[..., 3]):
            break
    else:
        raise RuntimeError("Text is not visible")
    meta[Keys.BLEND] = blend_modes
    return out


def _check_visibility(image, mask):
    gray = utils.to_gray(image[..., :3]).astype(np.uint8)
    mask = mask.astype(np.uint8)
    height, width = mask.shape

    peak = (mask > 127).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    bound = (mask > 0).astype(np.uint8)
    bound = cv2.dilate(bound, kernel, iterations=1)

    visit = bound.copy()
    visit ^= 1
    visit = np.pad(visit, 1, constant_values=1)

    border = bound.copy()
    border[mask > 0] = 0

    flag = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for y in range(height):
        for x in range(width):
            if peak[y][x]:
                cv2.floodFill(gray, visit, (x, y), 1, 16, 16, flag)

    visit = visit[1:-1, 1:-1]
    count = np.sum(visit & border)
    total = np.sum(border)
    return total > 0 and count <= total * 0.1


def _create_poly_mask(image, pad=0):
    height, width = image.shape[:2]
    alpha = image[..., 3].astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.float32)

    cts, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cts = sorted(cts, key=lambda ct: sum(cv2.boundingRect(ct)[:2]))

    if len(cts) == 1:
        hull = cv2.convexHull(cts[0])
        cv2.fillConvexPoly(mask, hull, 255)

    for idx in range(len(cts) - 1):
        pts = np.concatenate((cts[idx], cts[idx + 1]), axis=0)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    mask = utils.dilate_image(mask, pad)
    out = utils.create_image((width, height))
    out[..., 3] = mask
    return out


import pprint

import synthtiger
import json

config_path = "./examples/custom/config_kz_no_augment.yaml"
output_path = "./results/invest"
input_meta_path = "./tests/input_meta.json"

synthtiger.set_global_random_seed(seed=0)

config = synthtiger.read_config(config_path)
pprint.pprint(config)
template = SynthTiger(config)

# Open the JSON file for reading
with open(input_meta_path, 'r') as json_file:
    # Use json.load() to load the JSON data into a dictionary
    input_meta = json.load(json_file)
data = template.generate_from_meta(input_meta=input_meta)

template.init_save(output_path)
template.save(output_path, data, 0)
template.end_save(output_path)
print("End save")
