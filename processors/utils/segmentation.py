import cv2
import numpy as np


def make_image_from_words(words, width, height):
    words = sorted(words, key=lambda x: (x[1], x[0]))
    width = int(width)
    height = int(height)
    page_image_undilated = np.zeros((height, width), dtype=np.uint8)
    word_heights = []
    for word in words:
        x_0 = int(word[0])
        x_1 = int(word[2])
        y_0 = int(word[1])
        y_1 = int(word[3])
        word_heights.append(y_1 - y_0)
        page_image_undilated[y_0:y_1, x_0:x_1] = 255
    ker = np.ones((1, int(round(0.01 * width))), dtype=np.uint8)
    page_image = cv2.dilate(page_image_undilated, ker, iterations=1)
    word_heights = [int(x) for x in word_heights]
    median_height = np.median(word_heights)
    return page_image_undilated, page_image, median_height


def cut_segment(segment):
    binary = np.any(segment, axis=1).astype('uint8')
    _, contours, _ = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for contour in contours:
        points = [list(x) for xx in contour for x in xx]
        points = np.array(points)
        points = points[:, 1]
        if len(points) == 1:
            regions.append([points[0], points[0]+1])
        else:
            points[1] += 1
            regions.append(list(points))
    regions = sorted(regions, key=lambda x: (x[0]))
    return regions


def are_regions_nearby(box1, box2, words, median_height, threshold=0.75):
    prev_start, prev_stop = box1
    start, stop = box2
    prev_words = [w for w in words if prev_start <= w[1]
                  <= prev_stop or prev_start <= w[3] <= prev_stop]
    curr_words = [w for w in words if start <=
                  w[1] <= stop or start <= w[3] <= stop]
    prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
    curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
    prev_word = prev_words[-1]
    curr_word = curr_words[0]
    if (curr_word[1] - prev_word[3]) > threshold*median_height:
        return False
    return True


def is_alignment_similar(box1, box2, page_matrix):
    width = np.shape(page_matrix)[1]
    prev_image = page_matrix[box1[0]:box1[1], :]
    next_image = page_matrix[box2[0]:box2[1], :]
    prev_cuts = cut_segment(prev_image.T)
    next_cuts = cut_segment(next_image.T)
    if not prev_cuts or not next_cuts:
        return False
    if prev_cuts[0][0] < width // 2 and next_cuts[0][0] > width // 2:
        return False
    return True


def are_columns_separable(box, page_matrix):
    width = np.shape(page_matrix)[1]
    box_matrix = page_matrix[box[0]:box[1], :]
    ker = np.ones((1, int(round(0.01 * width))), dtype=np.uint8)
    box_dilated = cv2.dilate(box_matrix, ker, iterations=3)
    cuts = cut_segment(box_dilated.T)
    if not cuts:
        return False
    return bool(len(cuts) > 1)


def get_segment_margins(segment_matrix):
    top, bottom, left, right = [0]*4
    gaps_tb = list(np.any(segment_matrix, axis=1).astype('int'))
    gaps_lr = list(np.any(segment_matrix.T, axis=1).astype('int'))

    for i, val in enumerate(gaps_tb):
        if val == 1:
            break
    top = i

    gaps_tb.reverse()
    for i, val in enumerate(gaps_tb):
        if val == 1:
            break
    bottom = len(gaps_tb) - i

    for i, val in enumerate(gaps_lr):
        if val == 1:
            break
    left = i

    gaps_lr.reverse()
    for i, val in enumerate(gaps_lr):
        if val == 1:
            break
    right = len(gaps_lr) - i

    return top, bottom, left, right


def is_region_sparse(box, page_matrix, threshold=0.8):
    box_matrix = page_matrix[box[0]:box[1], :]
    margins = get_segment_margins(box_matrix)
    cropped_matrix = box_matrix[margins[0]:margins[1], margins[2]:margins[3]]
    num_white_pix = np.sum(cropped_matrix == 255)
    total_pix = cropped_matrix.shape[0]*cropped_matrix.shape[1]
    ratio = float(num_white_pix)/total_pix
    return bool(ratio <= threshold)


def label_segments(tb_cuts, lr_cuts, page_matrix):
    # Label segments as PARA or TABLE - First Pass
    segments = []
    for t_b, l_r in zip(tb_cuts, lr_cuts):
        num_cols = len(l_r)
        if (num_cols) > 1:
            if is_region_sparse(t_b, page_matrix):
                if are_columns_separable(t_b, page_matrix):
                    block_type = "TABLE"
                else:
                    block_type = "PARA"
            else:
                block_type = "PARA"
        elif num_cols == 1:
            block_type = "PARA"
        else:
            block_type = None
        if block_type:
            segments.append([t_b[0], t_b[1], block_type])
    return segments


def merge_table_neighbors(segments, page_matrix, words, median_height, threshold):
    processed = []
    merged_segments = []
    for idx, segment in enumerate(segments):
        merge = []
        if idx in processed:
            continue
        _, _, label = segment
        if label != "TABLE":
            continue
        prev = int(idx)
        while True:
            prev -= 1
            if prev < 0 or prev in processed:
                break
            prev_segment = segments[prev]
            prev_start, prev_stop, prev_label = prev_segment
            if prev_label == "TABLE":
                start, stop, label = segments[prev+1]
                is_nearby = are_regions_nearby([prev_start, prev_stop], [
                                               start, stop], words, median_height, threshold)
                if is_nearby:
                    merge.append(prev)
                    processed.append(prev)
                else:
                    break
            elif prev_label == "PARA":
                # Check distance, check alignment
                start, stop, label = segments[prev+1]
                is_nearby = are_regions_nearby([prev_start, prev_stop], [
                                               start, stop], words, median_height, threshold)
                if is_nearby:
                    tmp = page_matrix[prev_start:prev_stop, :]
                    tmp_width = np.shape(page_matrix)[1]
                    tmp_cuts = cut_segment(tmp.T)
                    if (tmp_cuts[0][0] < tmp_width // 2):
                        if (tmp_cuts[0][1] > tmp_width // 2):
                            break

                    # tmp = page_matrix[start:stop, :]
                    # tmp_cuts = cut_segment(tmp.T)
                    # left = tmp_cuts[0][0]
                    # right = tmp_cuts[-1][1]
                    # tmp = page_matrix[prev_start:prev_stop, left:right]
                    # binary = np.any(tmp, axis=0).astype("int")
                    # num_white_pix = np.sum(binary)
                    # num_pix = len(binary)
                    # ratio = float(num_white_pix)/num_pix
                    # if ratio > 0.75:
                    #     break

                    merge.append(prev)
                    processed.append(prev)
                else:
                    break

        _, _, label = segment
        nxt = int(idx)
        while True:
            nxt += 1
            if nxt >= len(segments) or nxt in processed:
                break
            nxt_segment = segments[nxt]
            nxt_start, nxt_stop, nxt_label = nxt_segment
            if nxt_label == "TABLE":
                start, stop, label = segments[nxt-1]
                is_nearby = are_regions_nearby(
                    [start, stop], [nxt_start, nxt_stop], words, median_height, threshold)
                if is_nearby:
                    merge.append(nxt)
                    processed.append(nxt)
                else:
                    break
            elif nxt_label == "PARA":
                # Check distance, check alignment
                start, stop, label = segments[nxt-1]
                is_nearby = are_regions_nearby(
                    [start, stop], [nxt_start, nxt_stop], words, median_height, threshold)
                if is_nearby:
                    tmp = page_matrix[nxt_start:nxt_stop, :]
                    tmp_width = np.shape(page_matrix)[1]
                    tmp_cuts = cut_segment(tmp.T)
                    if (tmp_cuts[0][0] < tmp_width // 2):
                        if (tmp_cuts[0][1] > tmp_width // 2):
                            break

                    # tmp = page_matrix[start:stop, :]
                    # tmp_cuts = cut_segment(tmp.T)
                    # left = tmp_cuts[0][0]
                    # right = tmp_cuts[-1][1]
                    # tmp = page_matrix[prev_start:prev_stop, left:right]
                    # binary = np.any(tmp, axis=0).astype("int")
                    # num_white_pix = np.sum(binary)
                    # num_pix = len(binary)
                    # ratio = float(num_white_pix)/num_pix
                    # if ratio > 0.75:
                    #     break

                    merge.append(nxt)
                    processed.append(nxt)
                else:
                    break

        if merge:
            merge.append(idx)
            merge.sort()
            processed.append(idx)
            merged_segments.append(merge)

    new_segments = []
    indices = []
    for seg in merged_segments:
        indices.extend(seg)
        start = segments[seg[0]][0]
        stop = segments[seg[-1]][1]
        new_segments.append([start, stop, "TABLE"])
    indices = list(set(indices))
    indices.sort()

    for idx, segment in enumerate(segments):
        if idx not in indices:
            new_segments.append(segment)

    segments = sorted(new_segments, key=lambda x: (x[0]))
    return segments


def merge_paragraph_neighbors(segments, page_matrix, words, median_height, threshold):
    processed = []
    merged_segments = []
    for idx, segment in enumerate(segments):
        merge = []
        if idx in processed:
            continue
        _, _, label = segment
        if label != "PARA":
            continue
        prev = int(idx)
        while True:
            prev -= 1
            if prev < 0 or prev in processed:
                break
            prev_segment = segments[prev]
            prev_start, prev_stop, prev_label = prev_segment
            if prev_label == "PARA":
                start, stop, label = segments[prev+1]
                # Check distance
                is_nearby = are_regions_nearby([prev_start, prev_stop], [
                                               start, stop], words, median_height, threshold)
                if is_nearby:
                    if is_alignment_similar([prev_start, prev_stop], [start, stop], page_matrix):
                        merge.append(prev)
                        processed.append(prev)
                    else:
                        break
                else:
                    break
            else:
                break
        nxt = int(idx)
        while True:
            nxt += 1
            if nxt >= len(segments) or nxt in processed:
                break
            nxt_segment = segments[nxt]
            nxt_start, nxt_stop, nxt_label = nxt_segment
            if nxt_label == "PARA":
                start, stop, label = segments[nxt-1]
                is_nearby = are_regions_nearby(
                    [start, stop], [nxt_start, nxt_stop], words, median_height, threshold)
                if is_nearby:
                    if is_alignment_similar([start, stop], [nxt_start, nxt_stop], page_matrix):
                        merge.append(nxt)
                        processed.append(nxt)
                else:
                    break
            else:
                break

        if merge:
            merge.append(idx)
            merge.sort()
            processed.append(idx)
            merged_segments.append(merge)

    new_segments = []
    indices = []
    for seg in merged_segments:
        indices.extend(seg)
        start = segments[seg[0]][0]
        stop = segments[seg[-1]][1]
        new_segments.append([start, stop, "PARA"])
    indices = list(set(indices))
    indices.sort()

    for idx, segment in enumerate(segments):
        if idx not in indices:
            new_segments.append(segment)

    segments = sorted(new_segments, key=lambda x: (x[0]))
    return segments


def are_tables_similar(segment1, segment2, tb_cuts, lr_cuts, height):
    start1, stop1, _ = segment1
    start2, stop2, _ = segment2

    gap = float(start2 - stop1)
    if gap > 0.05*height:
        return False

    rows1 = []
    for index, t_b in enumerate(tb_cuts):
        if t_b[0] < start1:
            continue
        if t_b[1] > stop1:
            continue
        rows1.append(index)

    rows2 = []
    for index, t_b in enumerate(tb_cuts):
        if t_b[0] < start2:
            continue
        if t_b[1] > stop2:
            continue
        rows2.append(index)

    cuts1 = [lr_cuts[i] for i in rows1]
    cuts2 = [lr_cuts[i] for i in rows2]

    if not cuts1 or not cuts2:
        return False

    left1 = min([x[0] for xx in cuts1 for x in xx])
    left2 = min([x[0] for xx in cuts2 for x in xx])
    right1 = max([x[1] for xx in cuts1 for x in xx])
    right2 = max([x[1] for xx in cuts2 for x in xx])

    if np.abs(left1 - left2) < 5 and np.abs(right1 - right2) < 5:
        return True

    return False


def merge_consecutive_tables(segments, tb_cuts, lr_cuts, page_matrix):
    processed = []
    merged_segments = []
    for idx, segment in enumerate(segments):
        merge = []
        if idx in processed:
            continue
        _, _, label = segment
        if label != "TABLE":
            continue
        nxt = int(idx)
        while True:
            nxt += 1
            if nxt >= len(segments) or nxt in processed:
                break
            nxt_segment = segments[nxt]
            _, _, nxt_label = nxt_segment
            if nxt_label != "TABLE":
                break
            is_similar = are_tables_similar(
                segments[nxt-1], nxt_segment, tb_cuts, lr_cuts, page_matrix.shape[0])
            if is_similar:
                merge.append(nxt)
                processed.append(nxt)
            else:
                break

        if merge:
            merge.append(idx)
            merge.sort()
            processed.append(idx)
            merged_segments.append(merge)

    new_segments = []
    indices = []
    for seg in merged_segments:
        indices.extend(seg)
        start = segments[seg[0]][0]
        stop = segments[seg[-1]][1]
        new_segments.append([start, stop, "TABLE"])
    indices = list(set(indices))
    indices.sort()

    for idx, segment in enumerate(segments):
        if idx not in indices:
            new_segments.append(segment)

    segments = sorted(new_segments, key=lambda x: (x[0]))
    return segments


def make_cells(segment, margins, words, start, stop):
    cropped_segment = segment[margins[0]:margins[1], margins[2]:margins[3]]
    tb_cuts = cut_segment(cropped_segment)
    lr_cuts = []
    max_len = 0
    max_idx = -1
    for idx, t_b in enumerate(tb_cuts):
        row_matrix = cropped_segment[t_b[0]:t_b[1], :].T
        l_r = cut_segment(row_matrix)
        lr_cuts.append(l_r)
        if len(l_r) >= max_len:
            max_len = len(l_r)
            max_idx = idx
    table = []
    if max_idx == -1:
        return table

    ref_cells = lr_cuts[max_idx]
    for idx, (t_b, l_r) in enumerate(zip(tb_cuts, lr_cuts)):
        x_0 = margins[2]
        x_1 = margins[3]
        y_0 = float(start + margins[0] + t_b[0])
        y_1 = float(start + margins[0] + t_b[1])
        row_words = [w for w in words if w[1] >= y_0]
        row_words = [w for w in row_words if w[3] <= y_1]
        row_words = [w for w in row_words if w[0] >= x_0]
        row_words = [w for w in row_words if w[2] <= x_1]
        row = []
        if len(l_r) == max_len:
            # Found exact match
            for cell_x_0, cell_x_1 in l_r:
                cell_words = [w for w in row_words if w[0] >= x_0 + cell_x_0]
                cell_words = [w for w in cell_words if w[2] <= x_0 + cell_x_1]
                cell_words = reorder_words(cell_words)
                row.append(cell_words)
            table.append(row)
        else:
            row = [[] for _ in range(max_len)]
            non_empty = False
            for cell_x_0, cell_x_1 in l_r:
                cell_words = [w for w in row_words if w[0] >= x_0 + cell_x_0]
                cell_words = [w for w in cell_words if w[2] <= x_0 + cell_x_1]
                if not cell_words:
                    continue
                cell_words = reorder_words(cell_words)
                cell_start = cell_words[0][0]
                cell_stop = cell_words[-1][2]
                selected_cell = None
                for cell_idx, tmp in enumerate(ref_cells):
                    if cell_start <= x_0 + tmp[1]:
                        selected_cell = cell_idx
                        break
                for cell_idx, tmp in enumerate(ref_cells):
                    if cell_stop <= x_0 + tmp[0]:
                        break
                if selected_cell is not None:
                    row[selected_cell] += cell_words
                    non_empty = True
            if non_empty:
                table.append(row)
    return table


def is_y_similar(item1, item2):
    _, y_0, _, y_1 = item1[0:4]
    _, yy_0, _, yy_1 = item2[0:4]
    d_0 = np.abs(y_0 - yy_0)
    d_1 = np.abs(y_1 - yy_1)
    if d_0 < 5 and d_1 < 5:
        return True
    return False


def reorder_words(words):
    words = sorted(words, key=lambda x: (x[1], x[0]))
    indices = []
    unique_y = {}
    for idx, item in enumerate(words):
        if idx not in indices:
            indices.append(idx)
            unique_y[tuple(item[0:4])] = [item]
            for idx2, item2 in enumerate(words):
                if idx2 not in indices:
                    y_similar = is_y_similar(item, item2)
                    if y_similar:
                        unique_y[tuple(item[0:4])].append(item2)
                        indices.append(idx2)
    new_words = []
    for _, value in unique_y.items():
        y_0 = min([x[1] for x in value])
        y_1 = max([x[3] for x in value])
        for x in value:
            new_words.append([x[0], y_0, x[2], y_1, x[-2], x[-1]])
    new_words = sorted(new_words, key=lambda x: (x[1], x[0]))
    return new_words


def make_paragraph(words, start, stop):
    block = None
    start = float(start) - 2
    stop = float(stop) + 2
    para_words = [w for w in words if w[1] >= start]
    para_words = [w for w in para_words if w[3] <= stop]
    para_words = sorted(para_words, key=lambda x: (x[1], x[0]))
    para_words = reorder_words(para_words)
    if para_words:
        points = [w[0:4] for w in para_words]
        points = np.array(points)
        x_0, y_0 = np.min(points[:, 0:2], axis=0)
        x_1, y_1 = np.max(points[:, 2:4], axis=0)
        return [x_0, y_0, x_1, y_1, "PARA", para_words]
    return block


def get_lines(cell_words):
    cell_words = sorted(cell_words, key=lambda x: (x[1], x[0]))
    indices = []
    unique_y = {}
    for idx, item in enumerate(cell_words):
        if idx not in indices:
            indices.append(idx)
            unique_y[tuple(item[0:4])] = [item]
            for idx2, item2 in enumerate(cell_words):
                if idx2 not in indices:
                    y_similar = is_y_similar(item, item2)
                    if y_similar:
                        unique_y[tuple(item[0:4])].append(item2)
                        indices.append(idx2)
    lines = []
    for _, value in unique_y.items():
        lines.append(sorted(value, key=lambda x: (x[1], x[0])))
    return lines


def split_multiline_cells(table):
    new_table = []
    for row in table:
        num_lines = []
        cols = []
        for cell in row:
            lines = get_lines(cell)
            cols.append(lines)
            num_lines.append(len(lines))
        unique = set(num_lines)
        if 0 in unique:
            unique.remove(0)
        if len(unique) == 1:
            if 1 in unique:
                new_table.append(row)
                continue
            else:
                num = list(unique)[0]
                for idx in range(num):
                    new_row = []
                    for cell in cols:
                        try:
                            new_row.append(cell[idx])
                        except:
                            new_row.append([])
                    new_table.append(new_row)
        else:
            max_lines_idx = np.argsort(num_lines)[-1]
            ref = cols[max_lines_idx]
            prev_rr = False
            p_rr = [[] for _ in range(len(cols))]
            rr = [[] for _ in range(len(cols))]
            for line in ref:
                y_0 = min([w[1] for w in line])
                y_1 = max([w[3] for w in line])
                rr[max_lines_idx].extend(line)
                for idx, col in enumerate(cols):
                    if idx == max_lines_idx:
                        continue
                    for line2 in col:
                        yy_0 = min([w[1] for w in line2])
                        yy_1 = max([w[3] for w in line2])
                        d_0 = np.abs(y_0 - yy_0)
                        d_1 = np.abs(y_1 - yy_1)
                        if (d_0 < 5) and (d_1 < 5):
                            rr[idx].extend(line2)
                ll = [float(len(xx) > 0) for xx in rr]
                m = np.mean(ll)
                if m <= 0.5:
                    if prev_rr is False:
                        new_table.append(rr)
                        prev_rr = False
                        p_rr = [[] for _ in range(len(cols))]
                        rr = [[] for _ in range(len(cols))]
                    else:
                        for j, tmp in enumerate(rr):
                            p_rr[j].extend(tmp)
                        prev_rr = True
                        rr = [[] for _ in range(len(cols))]
                else:
                    if prev_rr is not False:
                        new_table.append(p_rr)
                    p_rr = rr
                    prev_rr = True
                    rr = [[] for _ in range(len(cols))]
            if prev_rr is not False:
                new_table.append(p_rr)

    return new_table


def make_table(table):
    block = None
    table = split_multiline_cells(table)
    words = [x for row in table for cell in row for x in cell if x]
    words = sorted(words, key=lambda x: (x[1], x[0]))
    if words:
        points = [w[0:4] for w in words]
        points = np.array(points)
        x_0, y_0 = np.min(points[:, 0:2], axis=0)
        x_1, y_1 = np.max(points[:, 2:4], axis=0)
        return [x_0, y_0, x_1, y_1, "TABLE", table]
    return block


def make_blocks(segments, page_matrix, words):
    blocks = []
    for segment in segments:
        block = {}
        start, stop, label = segment
        if label == "TABLE":
            segment_matrix = page_matrix[start:stop, :]
            margins = get_segment_margins(segment_matrix)
            table = make_cells(segment_matrix, margins, words, start, stop)
            block = make_table(table)
        else:
            block = make_paragraph(words, start, stop)
        if block is not None:
            blocks.append(block)
    return blocks
