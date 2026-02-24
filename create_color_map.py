import os
import glob
import uuid
import re
import numpy as np
from osgeo import gdal

"""
Отдельный скрипт для создания цветной карты из монохромных GeoTIFF.

Что делает:
1) Ищет GeoTIFF в папке с мозаиками (по умолчанию WORK_DIR/mosaics)
2) (Опционально) маскирует каждый TIFF по полигону (cutline)
3) Преобразует значения в RGB по выбранной палитре
4) Для owiWindDirection использует фон от owiWindSpeed и рисует направления ветра стрелками
5) Добавляет ПОД картой отдельную панель-легенду (градиент + подписи значений)
6) Сохраняет цветной RGBA GeoTIFF в WORK_DIR/color_maps
"""

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

WORK_DIR = r"D:\WIND\data\OCN\119"
MOSAIC_DIR = "mosaics"
COLOR_OUTPUT_DIR = "color_maps"
INPUT_PATTERN = "*.tif"

# Маскирование результата по полигону
CLIP_BY_POLYGON = True
CLIP_POLYGON_PATH = r"D:\WIND\data\ЯпонскоеМоре-буффер-5км.gpkg"
CLIP_LAYER_NAME = None
CLIP_EXCLUDE_VARIABLES = {"owiEcmwfWindSpeed"}  # исключения из маскирования по полигону

# Палитры: 'jet', 'turbo', 'viridis'
COLORMAP = "turbo"

# Диапазон данных
AUTO_RANGE = True
PERCENTILE_MIN = 2
PERCENTILE_MAX = 98
GLOBAL_MIN = 0.0
GLOBAL_MAX = 35.0

# Легенда
ADD_LEGEND_PANEL = True
LEGEND_PANEL_HEIGHT_PX = 64
LEGEND_HORIZONTAL_MARGIN_PX = 24
LEGEND_BAR_HEIGHT_PX = 18
LEGEND_TICK_COUNT = 5
CREATE_LEGEND_TXT = True

# Убирать пустые края (NoData) по фактическим пикселям результата
TRIM_EMPTY_BORDERS = True

# Настройки стрелок ветра (для owiWindDirection)
DRAW_WIND_ARROWS = True
ARROW_STEP_PX = 40
ARROW_LENGTH_PX = 16
ARROW_COLOR = (0, 0, 0)
ARROW_HEAD_SIZE_PX = 5

# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Создана папка: {path}")


def get_colormap(name: str = "turbo"):
    name = name.lower()
    if name == "viridis":
        anchors = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]
    elif name == "jet":
        anchors = [(0, 0, 131), (0, 60, 170), (5, 255, 255), (255, 255, 0), (250, 0, 0), (128, 0, 0)]
    else:
        anchors = [(48, 18, 59), (50, 87, 184), (29, 150, 222), (33, 204, 145), (170, 220, 50), (245, 171, 35), (236, 76, 61), (122, 4, 3)]

    anchors = np.array(anchors, dtype=np.float32)
    x = np.linspace(0, 1, anchors.shape[0])
    xi = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for c in range(3):
        lut[:, c] = np.interp(xi, x, anchors[:, c]).astype(np.uint8)
    return lut


def normalize_to_uint8(data: np.ndarray, data_min: float, data_max: float):
    if np.isclose(data_max, data_min):
        return np.zeros_like(data, dtype=np.uint8)
    norm = (data - data_min) / (data_max - data_min)
    norm = np.clip(norm, 0, 1)
    return (norm * 255).astype(np.uint8)


FONT_5X7 = {
    "0": ["11111", "10001", "10001", "10001", "10001", "10001", "11111"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["11111", "00001", "00001", "11111", "10000", "10000", "11111"],
    "3": ["11111", "00001", "00001", "01111", "00001", "00001", "11111"],
    "4": ["10001", "10001", "10001", "11111", "00001", "00001", "00001"],
    "5": ["11111", "10000", "10000", "11111", "00001", "00001", "11111"],
    "6": ["11111", "10000", "10000", "11111", "10001", "10001", "11111"],
    "7": ["11111", "00001", "00010", "00100", "01000", "10000", "10000"],
    "8": ["11111", "10001", "10001", "11111", "10001", "10001", "11111"],
    "9": ["11111", "10001", "10001", "11111", "00001", "00001", "11111"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def draw_char_5x7(canvas, x, y, ch, color=(0, 0, 0), scale=1):
    glyph = FONT_5X7.get(ch, FONT_5X7[" "])
    h, w, _ = canvas.shape
    for gy, row in enumerate(glyph):
        for gx, bit in enumerate(row):
            if bit == "1":
                x0, y0 = x + gx * scale, y + gy * scale
                x1, y1 = x0 + scale, y0 + scale
                if x0 < w and y0 < h and x1 > 0 and y1 > 0:
                    canvas[max(y0, 0):min(y1, h), max(x0, 0):min(x1, w), :] = color


def draw_text_5x7(canvas, x, y, text, color=(0, 0, 0), scale=1):
    cursor = x
    for ch in text:
        draw_char_5x7(canvas, cursor, y, ch, color=color, scale=scale)
        cursor += 6 * scale


def format_tick(value: float) -> str:
    return f"{value:.2f}"


def trim_to_valid_pixels(data: np.ndarray, valid_mask: np.ndarray, geotransform):
    ys, xs = np.where(valid_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    data_trim = data[y_min:y_max + 1, x_min:x_max + 1]
    mask_trim = valid_mask[y_min:y_max + 1, x_min:x_max + 1]

    gt_new = geotransform
    if geotransform is not None:
        gt = geotransform
        new_x0 = gt[0] + x_min * gt[1] + y_min * gt[2]
        new_y0 = gt[3] + x_min * gt[4] + y_min * gt[5]
        gt_new = (new_x0, gt[1], gt[2], new_y0, gt[4], gt[5])
    return data_trim, mask_trim, gt_new


def trim_pair_to_common_mask(data_a: np.ndarray, data_b: np.ndarray, common_mask: np.ndarray, geotransform):
    """Обрезает два массива и общую маску по bbox common_mask, с обновлением geotransform."""
    ys, xs = np.where(common_mask)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    a_trim = data_a[y_min:y_max + 1, x_min:x_max + 1]
    b_trim = data_b[y_min:y_max + 1, x_min:x_max + 1]
    m_trim = common_mask[y_min:y_max + 1, x_min:x_max + 1]

    gt_new = geotransform
    if geotransform is not None:
        gt = geotransform
        new_x0 = gt[0] + x_min * gt[1] + y_min * gt[2]
        new_y0 = gt[3] + x_min * gt[4] + y_min * gt[5]
        gt_new = (new_x0, gt[1], gt[2], new_y0, gt[4], gt[5])

    return a_trim, b_trim, m_trim, gt_new


def clip_raster_to_polygon(input_tif: str, temp_dir: str) -> str:
    src_ds = gdal.Open(input_tif, gdal.GA_ReadOnly)
    if src_ds is None:
        raise RuntimeError(f"Не удалось открыть {input_tif}")
    nodata = src_ds.GetRasterBand(1).GetNoDataValue()
    src_ds = None

    temp_path = os.path.join(temp_dir, f"clipped_{uuid.uuid4().hex}.tif")
    warp_kwargs = {
        "format": "GTiff",
        "cutlineDSName": CLIP_POLYGON_PATH,
        "cropToCutline": False,
        "multithread": True,
        "creationOptions": ["COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    }
    if CLIP_LAYER_NAME:
        warp_kwargs["cutlineLayer"] = CLIP_LAYER_NAME
    if nodata is not None:
        warp_kwargs["srcNodata"] = nodata
        warp_kwargs["dstNodata"] = nodata

    out_ds = gdal.Warp(temp_path, input_tif, options=gdal.WarpOptions(**warp_kwargs))
    if out_ds is None:
        raise RuntimeError(f"Ошибка маскирования по полигону: {input_tif}")
    out_ds = None
    return temp_path


def should_clip_file(input_tif: str) -> bool:
    """Определяет, нужно ли применять маскирование по полигону для конкретного файла."""
    base = os.path.splitext(os.path.basename(input_tif))[0]
    var, _ = split_var_and_date(base)
    if var in CLIP_EXCLUDE_VARIABLES:
        return False
    return True


def apply_clip_if_enabled(input_tif: str, temp_dir: str):
    if CLIP_BY_POLYGON and should_clip_file(input_tif):
        if not os.path.exists(CLIP_POLYGON_PATH):
            raise RuntimeError(f"Файл полигона не найден: {CLIP_POLYGON_PATH}")
        clipped = clip_raster_to_polygon(input_tif, temp_dir)
        return clipped, clipped
    return input_tif, None


def append_legend_panel(rgb: np.ndarray, alpha: np.ndarray, lut: np.ndarray, dmin: float, dmax: float):
    h, w, _ = rgb.shape
    panel_h = max(40, int(LEGEND_PANEL_HEIGHT_PX))
    margin_x = max(8, int(LEGEND_HORIZONTAL_MARGIN_PX))
    bar_h = max(8, min(panel_h // 2, int(LEGEND_BAR_HEIGHT_PX)))

    new_h = h + panel_h
    out_rgb = np.full((new_h, w, 3), 255, dtype=np.uint8)
    out_alpha = np.full((new_h, w), 255, dtype=np.uint8)
    out_rgb[:h, :, :] = rgb
    out_alpha[:h, :] = alpha

    panel_y0 = h
    bar_y0 = panel_y0 + 8
    bar_y1 = min(panel_y0 + panel_h - 20, bar_y0 + bar_h)
    x0, x1 = margin_x, max(margin_x + 10, w - margin_x)
    bar_w = x1 - x0

    idx = np.linspace(0, 255, bar_w).astype(np.uint8)
    out_rgb[bar_y0:bar_y1, x0:x1, :] = lut[idx][np.newaxis, :, :]

    out_rgb[bar_y0:bar_y0 + 1, x0:x1, :] = 0
    out_rgb[bar_y1 - 1:bar_y1, x0:x1, :] = 0
    out_rgb[bar_y0:bar_y1, x0:x0 + 1, :] = 0
    out_rgb[bar_y0:bar_y1, x1 - 1:x1, :] = 0

    tick_count = max(2, int(LEGEND_TICK_COUNT))
    tick_x = np.linspace(x0, x1 - 1, tick_count).astype(int)
    tick_vals = np.linspace(dmin, dmax, tick_count)
    label_y = bar_y1 + 4

    for tx, tv in zip(tick_x, tick_vals):
        out_rgb[bar_y1:min(panel_y0 + panel_h, bar_y1 + 6), tx:tx + 2, :] = 0
        label = format_tick(float(tv))
        label_w = len(label) * 6
        lx = max(0, min(int(tx - label_w // 2), w - label_w))
        draw_text_5x7(out_rgb, lx, label_y, label, color=(0, 0, 0), scale=1)

    return out_rgb, out_alpha


def draw_line(rgb: np.ndarray, x0: int, y0: int, x1: int, y1: int, color):
    h, w, _ = rgb.shape
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    for i in range(steps + 1):
        x = int(round(x0 + dx * (i / steps)))
        y = int(round(y0 + dy * (i / steps)))
        if 0 <= x < w and 0 <= y < h:
            rgb[y, x, :] = color


def draw_arrow(rgb: np.ndarray, x: int, y: int, angle_rad: float, length: int, color):
    x2 = int(round(x + length * np.cos(angle_rad)))
    y2 = int(round(y - length * np.sin(angle_rad)))
    draw_line(rgb, x, y, x2, y2, color)

    head = max(3, int(ARROW_HEAD_SIZE_PX))
    a1 = angle_rad + np.deg2rad(150)
    a2 = angle_rad - np.deg2rad(150)
    hx1 = int(round(x2 + head * np.cos(a1)))
    hy1 = int(round(y2 - head * np.sin(a1)))
    hx2 = int(round(x2 + head * np.cos(a2)))
    hy2 = int(round(y2 - head * np.sin(a2)))
    draw_line(rgb, x2, y2, hx1, hy1, color)
    draw_line(rgb, x2, y2, hx2, hy2, color)


def overlay_wind_direction_arrows(rgb: np.ndarray, direction_data: np.ndarray, valid_mask: np.ndarray):
    if not DRAW_WIND_ARROWS:
        return
    h, w = direction_data.shape
    step = max(10, int(ARROW_STEP_PX))
    length = max(8, int(ARROW_LENGTH_PX))

    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not valid_mask[y, x]:
                continue
            d = direction_data[y, x]
            if not np.isfinite(d):
                continue
            # Метео-угол (откуда дует, градусы по часовой от севера) -> экранный угол
            angle = np.deg2rad(270.0 - float(d))
            draw_arrow(rgb, x, y, angle, length, ARROW_COLOR)


def create_legend_txt(output_txt: str, dmin: float, dmax: float, extra: str = ""):
    ticks = np.linspace(dmin, dmax, 5)
    lines = [
        "Legend (gradient panel below map)",
        f"min={dmin:.6f}",
        f"max={dmax:.6f}",
        "ticks=" + ", ".join([f"{t:.6f}" for t in ticks]),
    ]
    if extra:
        lines.append(extra)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def prepare_data_for_render(input_tif: str, temp_dir: str, do_trim: bool = True):
    working_tif, temp_file = apply_clip_if_enabled(input_tif, temp_dir)
    src = gdal.Open(working_tif, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Не удалось открыть {working_tif}")

    band = src.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    src_gt = src.GetGeoTransform(can_return_null=True)
    src_proj = src.GetProjection()

    valid_mask = np.isfinite(data)
    if nodata is not None:
        valid_mask &= (data != nodata)

    if not np.any(valid_mask):
        src = None
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
        raise RuntimeError(f"Нет валидных данных в {working_tif}")

    if TRIM_EMPTY_BORDERS and do_trim:
        data, valid_mask, src_gt = trim_to_valid_pixels(data, valid_mask, src_gt)

    src = None
    return data, valid_mask, src_gt, src_proj, temp_file


def render_color_from_data(data: np.ndarray, valid_mask: np.ndarray, lut: np.ndarray):
    if AUTO_RANGE:
        vals = data[valid_mask]
        dmin = float(np.percentile(vals, PERCENTILE_MIN))
        dmax = float(np.percentile(vals, PERCENTILE_MAX))
    else:
        dmin, dmax = GLOBAL_MIN, GLOBAL_MAX

    idx = np.zeros_like(data, dtype=np.uint8)
    idx[valid_mask] = normalize_to_uint8(data[valid_mask], dmin, dmax)
    rgb = lut[idx]
    alpha = np.zeros_like(idx, dtype=np.uint8)
    alpha[valid_mask] = 255
    return rgb, alpha, dmin, dmax


def split_var_and_date(base: str):
    m = re.match(r"^(?P<var>.+?)_(?P<date>\d{8})$", base)
    if m:
        return m.group("var"), m.group("date")
    return None, None


def format_output_base_name(input_base: str) -> str:
    m = re.match(r"^(?P<prefix>.+?)_(?P<date>\d{8})$", input_base)
    if m:
        return f"{m.group('date')}_{m.group('prefix')}"
    return input_base


def create_color_geotiff(input_tif: str, output_tif: str, lut: np.ndarray, temp_dir: str):
    base = os.path.splitext(os.path.basename(input_tif))[0]
    var, date = split_var_and_date(base)

    legend_extra = ""

    if var == "owiWindDirection":
        speed_base = f"owiWindSpeed_{date}" if date else None
        speed_tif = os.path.join(os.path.dirname(input_tif), f"{speed_base}.tif") if speed_base else None

        if speed_tif and os.path.exists(speed_tif):
            # Для согласования стрелок и фона читаем БЕЗ индивидуального trim,
            # затем обрезаем по общей маске.
            speed_data, speed_mask, src_gt, src_proj, temp_speed = prepare_data_for_render(speed_tif, temp_dir, do_trim=False)
            dir_data, dir_mask, _, _, temp_dir_file = prepare_data_for_render(input_tif, temp_dir, do_trim=False)

            if dir_data.shape == speed_data.shape:
                common_mask = speed_mask & dir_mask

                if not np.any(common_mask):
                    print(f"Предупреждение: нет общих валидных пикселей speed/direction для {base}, стрелки пропущены")
                    common_mask = speed_mask

                if TRIM_EMPTY_BORDERS and np.any(common_mask):
                    speed_data, dir_data, common_mask, src_gt = trim_pair_to_common_mask(
                        speed_data, dir_data, common_mask, src_gt
                    )

                rgb, alpha, dmin, dmax = render_color_from_data(speed_data, common_mask, lut)
                overlay_wind_direction_arrows(rgb, dir_data, common_mask)
                legend_extra = "overlay=wind_direction_arrows; background=owiWindSpeed"
            else:
                print(f"Предупреждение: размерности direction/speed отличаются для {base}, стрелки пропущены")
                rgb, alpha, dmin, dmax = render_color_from_data(speed_data, speed_mask, lut)

            for tf in [temp_speed, temp_dir_file]:
                if tf and os.path.exists(tf):
                    os.remove(tf)
        else:
            print(f"Предупреждение: не найден соответствующий owiWindSpeed для {base}; используем только direction")
            data, valid_mask, src_gt, src_proj, temp_file = prepare_data_for_render(input_tif, temp_dir)
            rgb, alpha, dmin, dmax = render_color_from_data(data, valid_mask, lut)
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        data, valid_mask, src_gt, src_proj, temp_file = prepare_data_for_render(input_tif, temp_dir)
        rgb, alpha, dmin, dmax = render_color_from_data(data, valid_mask, lut)
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

    if ADD_LEGEND_PANEL:
        rgb, alpha = append_legend_panel(rgb, alpha, lut, dmin, dmax)

    out_h, out_w, _ = rgb.shape
    dst = gdal.GetDriverByName("GTiff").Create(
        output_tif, out_w, out_h, 4, gdal.GDT_Byte,
        options=["COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"],
    )
    if dst is None:
        raise RuntimeError(f"Не удалось создать {output_tif}")

    if src_gt is not None:
        dst.SetGeoTransform(src_gt)
    if src_proj:
        dst.SetProjection(src_proj)

    dst.GetRasterBand(1).WriteArray(rgb[:, :, 0]); dst.GetRasterBand(1).SetDescription("Red")
    dst.GetRasterBand(2).WriteArray(rgb[:, :, 1]); dst.GetRasterBand(2).SetDescription("Green")
    dst.GetRasterBand(3).WriteArray(rgb[:, :, 2]); dst.GetRasterBand(3).SetDescription("Blue")
    dst.GetRasterBand(4).WriteArray(alpha); dst.GetRasterBand(4).SetDescription("Alpha")
    dst.GetRasterBand(4).SetNoDataValue(0)
    dst.FlushCache(); dst = None

    print(f"Готово: {os.path.basename(output_tif)} | диапазон [{dmin:.3f}; {dmax:.3f}] | палитра: {COLORMAP}")
    return dmin, dmax, legend_extra


def main():
    mosaic_abs = os.path.join(WORK_DIR, MOSAIC_DIR)
    output_abs = os.path.join(WORK_DIR, COLOR_OUTPUT_DIR)
    temp_abs = os.path.join(output_abs, "_temp")

    ensure_dir(output_abs)
    ensure_dir(temp_abs)

    input_files = sorted(glob.glob(os.path.join(mosaic_abs, INPUT_PATTERN)))
    if not input_files:
        print(f"В папке {mosaic_abs} не найдено файлов по шаблону {INPUT_PATTERN}")
        return

    lut = get_colormap(COLORMAP)

    print("=" * 70)
    print("Создание цветных карт из мозаик")
    print(f"Вход:  {mosaic_abs}")
    print(f"Выход: {output_abs}")
    print(f"Маскирование по полигону: {CLIP_BY_POLYGON}")
    print(f"Обрезка пустых краёв: {TRIM_EMPTY_BORDERS}")
    print(f"Стрелки направления ветра: {DRAW_WIND_ARROWS}")
    if CLIP_BY_POLYGON:
        print(f"Полигон: {CLIP_POLYGON_PATH}")
        print(f"Исключения из маскирования: {sorted(CLIP_EXCLUDE_VARIABLES)}")
    print(f"Файлов: {len(input_files)}")
    print("=" * 70)

    ok = 0
    for input_tif in input_files:
        base = os.path.splitext(os.path.basename(input_tif))[0]
        output_base = format_output_base_name(base)
        output_tif = os.path.join(output_abs, f"{output_base}_color.tif")
        try:
            dmin, dmax, legend_extra = create_color_geotiff(input_tif, output_tif, lut, temp_abs)
            if CREATE_LEGEND_TXT:
                legend_txt = os.path.join(output_abs, f"{output_base}_legend.txt")
                create_legend_txt(legend_txt, dmin, dmax, extra=legend_extra)
            ok += 1
        except Exception as e:
            print(f"Ошибка: {os.path.basename(input_tif)} -> {e}")

    if os.path.isdir(temp_abs) and not os.listdir(temp_abs):
        os.rmdir(temp_abs)

    print("=" * 70)
    print(f"Готово: {ok} из {len(input_files)}")
    print(f"Цветные карты: {output_abs}")
    print("=" * 70)


if __name__ == "__main__":
    main()
