import numpy as np
import glob
import tqdm
import cv2
import json
import pathlib


def parse_polygon(coordinates: dict, image_size: tuple) -> np.ndarray:
    mask = np.zeros(image_size, dtype=np.float32)
    if len(coordinates) == 1:
        points = [np.int32(coordinates)]
        cv2.fillPoly(mask, points, 1)
    else:
        points = [np.int32([coordinates[0]])]
        cv2.fillPoly(mask, points, 1)

        for polygon in coordinates[1:]:
            points = [np.int32([polygon])]
            cv2.fillPoly(mask, points, 0)
    return mask


def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
    """
    Метод для парсинга фигур из geojson файла
    """
    mask = np.zeros(image_size, dtype=np.float32)
    coordinates = shape['coordinates']
    if shape['type'] == 'MultiPolygon':
        for polygon in coordinates:
            mask += parse_polygon(polygon, image_size)
    else:
        mask += parse_polygon(coordinates, image_size)

    return mask

def read_layout(path: str, image_size: tuple) -> np.ndarray:
    with open(path, 'r', encoding='cp1251') as f: 
        json_contents = json.load(f)

    num_channels = 1 + max(CLASS_IDS.values())
    mask_channels = [np.zeros(image_size, dtype=np.float32) for _ in range(num_channels)]
    mask = np.zeros(image_size, dtype=np.float32)

    if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
        features = json_contents['features']
    elif type(json_contents) == list:
        features = json_contents
    else:
        features = [json_contents]

    for shape in features:
        channel_id = CLASS_IDS["vessel"]
        mask = parse_mask(shape['geometry'], image_size)
        mask_channels[channel_id] = np.maximum(mask_channels[channel_id], mask)

    mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)

    return np.stack(mask_channels, axis=-1)

def read_image(path: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    image = np.array(image / 255, dtype=np.float32)
    return image


if __name__ == '__main__':
    CLASS_IDS = {"vessel": 1}
    data_folder = pathlib.Path(r"..\Data")
    filtred_imgs_folder = data_folder / "train_filtred_dataset_mc\imgs"
    filtred_geojson_folder = data_folder / "train_filtred_dataset_mc\geojson"
    filtred_masks_folder = data_folder / "train_filtred_dataset_mc\masks"

    if not filtred_masks_folder.exists():
        filtred_masks_folder.mkdir()

    for geojson in tqdm.tqdm(filtred_geojson_folder.glob("*.geojson")):
        image = read_image(filtred_imgs_folder.joinpath(geojson.stem + '.png'))
        mask = read_layout(geojson, image.shape[:2])
        mask_indx = np.argmax(mask, axis=-1)
        mask_indx = np.expand_dims(mask_indx, axis=-1)
        cv2.imwrite(str(filtred_masks_folder.joinpath(geojson.stem + '.png')), np.uint8(mask_indx*255),  [cv2.IMWRITE_PNG_COMPRESSION, 9])



