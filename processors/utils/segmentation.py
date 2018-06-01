import cv2
import numpy as np


def image_from_words(words, width, height):
    words = sorted(words, key=lambda x: (x[1], x[0]))
    width = int(width)
    height = int(height)
    page_image_undilated = np.zeros((height, width), dtype=np.uint8)
    word_heights = []
    for word in words:
        x0 = int(word[0])
        x1 = int(word[2])
        y0 = int(word[1])
        y1 = int(word[3])
        word_heights.append(y1-y0)
        page_image_undilated[y0:y1, x0:x1] = 255
    ker = np.ones((1, int(round(0.01 * width))), dtype=np.uint8)
    page_image = cv2.dilate(page_image_undilated, ker, iterations=1)
    word_heights = [int(x) for x in word_heights]
    median_height = np.median(word_heights)
    return page_image_undilated, page_image, median_height


def cut_segment(segment):
    gaps = np.any(segment, axis=1).astype('int')
    cuts = []
    prev = 0
    end = 0
    for idx, curr in enumerate(gaps):
        if curr == 1:
            if prev == 0:
                cuts.append(idx)
        if curr == 0:
            if prev == 1:
                end = idx
        prev = curr
    cuts.append(end)
    return cuts


def label_segments(tb_cuts, lr_cuts, words, width, height, median_height):
    # Label segments as PARA or TABLE
    segments = []
    block_start = -1
    block_stop = -1
    block_type = None
    for idx, (tb, lr) in enumerate(zip(tb_cuts, lr_cuts)):
        col_count = len(lr)
        if col_count > 1:
            if block_start < 0:
                block_start = tb[0]
                block_type = "TABLE"
            else:
                if block_type != "TABLE":
                    block_stop = tb[0]
                    # Append Segment as PARA
                    segments.append([block_start, block_stop, "PARA"])
                    block_start = tb[0]
                    block_type = "TABLE"
                else:
                    prev_start, prev_stop = tb_cuts[idx-1]
                    prev_words = [w for w in words if prev_start <= w[1]
                                  <= prev_stop or prev_start <= w[3] <= prev_stop]
                    start, stop = tb
                    curr_words = [w for w in words if start <=
                                  w[1] <= stop or start <= w[3] <= stop]
                    prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
                    curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
                    prev_word = prev_words[-1]
                    curr_word = curr_words[0]
                    if (curr_word[1] - prev_word[3]) > 0.75*median_height:
                        block_stop = tb[0]
                        # Append Segment as TABLE
                        segments.append([block_start, block_stop, "TABLE"])
                        block_start = tb[0]
                        block_type = "TABLE"
        else:
            # TODO: Segregate paragraphs based on their alignments
            if block_start < 0:
                block_start = tb[0]
                block_type = "PARA"
            else:
                if block_type != "PARA":
                    block_stop = tb[0]
                    # Append Segment as TABLE
                    segments.append([block_start, block_stop, "TABLE"])
                    block_start = tb[0]
                    block_type = "PARA"
                else:
                    prev_start, prev_stop = tb_cuts[idx-1]
                    prev_words = [w for w in words if prev_start <= w[1]
                                  <= prev_stop or prev_start <= w[3] <= prev_stop]
                    start, stop = tb
                    curr_words = [w for w in words if start <=
                                  w[1] <= stop or start <= w[3] <= stop]
                    prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
                    curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
                    prev_word = prev_words[-1]
                    curr_word = curr_words[0]
                    if (curr_word[1] - prev_word[3]) > 0.75*median_height:
                        block_stop = tb[0]
                        # Append Segment as PARA
                        segments.append([block_start, block_stop, "PARA"])
                        block_start = tb[0]
                        block_type = "PARA"
    if block_start > -1:
        block_stop = tb[0]
        if block_start != block_stop:
            segments.append([block_start, block_stop, block_type])
    return segments


def merge_segments(segments, page_matrix, words, width, height, median_height):
    # Merge neighboring segments (applies to TABLE)
    merge_segments = []
    merge_list = []
    for idx, segment in enumerate(segments):
        merge = []
        if idx == 0:
            continue
        label = segment[-1]
        if label != "TABLE":
            continue
        prev = int(idx)
        while True:
            prev = prev - 1
            if prev < 0:
                break
            if prev in merge_list:
                break
            prev_segment = segments[prev]
            prev_label = prev_segment[-1]
            if prev_label == "TABLE":
                break
            prev_image = page_matrix[prev_segment[0]:prev_segment[1], :]
            prev_width = np.shape(prev_image)[1]
            cuts = cut_segment(prev_image.T)
            if cuts[0] < prev_width // 2:
                if cuts[1] > prev_width // 2:
                    break
            prev_start = segments[prev][0]
            prev_stop = segments[prev][1]
            prev_words = [w for w in words if prev_start <= w[1]
                          <= prev_stop or prev_start <= w[3] <= prev_stop]
            start = segments[prev+1][0]
            stop = segments[prev+1][1]
            curr_words = [w for w in words if start <=
                          w[1] <= stop or start <= w[3] <= stop]
            prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
            curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
            prev_word = prev_words[-1]
            curr_word = curr_words[0]
            if (curr_word[1] - prev_word[3]) > 0.75*median_height:
                break
            merge.append(prev)
            merge_list.append(prev)
        nxt = int(idx)
        while True:
            nxt = nxt + 1
            if nxt == len(segments):
                break
            if nxt in merge_list:
                break
            nxt_segment = segments[nxt]
            nxt_label = nxt_segment[-1]
            if nxt_label == "TABLE":
                break
            nxt_image = page_matrix[nxt_segment[0]:nxt_segment[1], :]
            nxt_width = np.shape(nxt_image)[1]
            cuts = cut_segment(nxt_image.T)
            if cuts[0] < nxt_width // 2:
                if cuts[1] > nxt_width // 2:
                    break
            prev_start = segments[nxt-1][0]
            prev_stop = segments[nxt-1][1]
            prev_words = [w for w in words if prev_start <= w[1]
                          <= prev_stop or prev_start <= w[3] <= prev_stop]
            start = segments[nxt][0]
            stop = segments[nxt][1]
            curr_words = [w for w in words if start <=
                          w[1] <= stop or start <= w[3] <= stop]
            prev_words = sorted(prev_words, key=lambda x: (x[1], x[0]))
            curr_words = sorted(curr_words, key=lambda x: (x[1], x[0]))
            prev_word = prev_words[-1]
            curr_word = curr_words[0]
            if (curr_word[1] - prev_word[3]) > 0.75*median_height:
                break
            merge.append(nxt)
            merge_list.append(nxt)
        if len(merge) > 0:
            merge.append(idx)
            merge.sort()
            merge_list.append(idx)
            merge_segments.append(merge)

    new_segments = []
    indices = []
    for m in merge_segments:
        indices.extend(m)
        m1 = m[0]
        s1 = segments[m1][0]
        m2 = m[-1]
        s2 = segments[m2][1]
        new_segments.append([s1, s2, "TABLE"])
    indices = list(set(indices))
    indices.sort()

    for idx, s in enumerate(segments):
        if idx not in indices:
            new_segments.append(s)

    segments = sorted(new_segments, key=lambda x: (x[0]))
    return segments


def merge_consecutive_tables(segments):
    merge_tables = []
    merge_list = []
    for idx, segment in enumerate(segments):
        if idx not in merge_list:
            merge = []
            label = segment[-1]
            if label != "TABLE":
                continue
            nxt = int(idx)
            while True:
                nxt = nxt + 1
                if nxt >= len(segments):
                    break
                nxt_label = segments[nxt][-1]
                if nxt_label != "TABLE":
                    break
                merge.append(nxt)
                merge_list.append(nxt)
            if len(merge) > 0:
                merge.append(idx)
                merge_list.append(idx)
                merge.sort()
                merge_tables.append(merge)
    new_segments = []
    indices = []
    for m in merge_tables:
        indices.extend(m)
        m1 = m[0]
        s1 = segments[m1][0]
        m2 = m[-1]
        s2 = segments[m2][1]
        new_segments.append([s1, s2, "TABLE"])
    indices = list(set(indices))
    indices.sort()

    for idx, s in enumerate(segments):
        if idx not in indices:
            new_segments.append(s)

    segments = sorted(new_segments, key=lambda x: (x[0]))
    return segments


def get_segment_margins(segment_matrix):
    top, bottom, left, right = [0]*4
    segment_matrix_T = np.transpose(segment_matrix)
    gaps = np.any(segment_matrix, axis=1).astype('int')
    gaps_T = np.any(segment_matrix_T, axis=1).astype('int')
    gaps = list(gaps)
    gaps_T = list(gaps_T)

    for i, val in enumerate(gaps):
        if val == 1:
            break
    top = i

    gaps.reverse()
    for i, val in enumerate(gaps):
        if val == 1:
            break
    bottom = len(gaps) - i

    for i, val in enumerate(gaps_T):
        if val == 1:
            break
    left = i

    gaps_T.reverse()
    for i, val in enumerate(gaps_T):
        if val == 1:
            break
    right = len(gaps_T) - i

    return top, bottom, left, right


def make_cells(segment, margins, words, start, stop):
    tmp1 = cut_segment(segment)
    tmp2 = tmp1[1:] + [np.shape(segment)[0]]
    tb_cuts = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
    lr_cuts = []
    max_len = 0
    max_idx = -1
    for idx, tb in enumerate(tb_cuts):
        line_matrix = segment[tb[0]:tb[1], :].T
        tmp1 = cut_segment(line_matrix)
        tmp2 = tmp1[1:] + [np.shape(line_matrix.T)[1]]
        tmp = [(t1, t2) for t1, t2 in zip(tmp1, tmp2)]
        lr_cuts.append(tmp)
        if len(tmp) >= max_len:
            max_len = len(tmp)
            max_idx = idx
    table = []
    if max_idx > -1:
        num_cols = max_len
        ref_start, ref_stop = tb_cuts[max_idx]
        y0 = float(start + margins[0] + ref_start) - 2
        y1 = float(y0 + (ref_stop - ref_start)) + 4
        W = [w for w in words if w[1] >= y0]
        W = [w for w in W if w[3] <= y1]
        ref_line_words = W
        ref_cuts = lr_cuts[max_idx][0:-1]
        ref_cells = []
        for seg in ref_cuts:
            x0 = float(seg[0]) - 2
            x1 = float(seg[1]) + 2
            W = [w for w in ref_line_words if w[0] >= x0]
            W = [w for w in W if w[2] <= x1]
            W = sorted(W, key=lambda x: (x[0]))
            ref_cells.append([W[0][0], W[-1][2]])
        for idx, (v, h) in enumerate(zip(tb_cuts, lr_cuts)):
            y0 = float(start + margins[0] + v[0]) - 2
            y1 = float(y0 + (v[1] - v[0])) + 4
            W = [w for w in words if w[1] >= y0]
            W = [w for w in W if w[3] <= y1]
            line_words = W
            row = []
            l = len(h)
            if l == max_len:
                # Found exact match
                for seg in h[0:-1]:
                    x0 = float(seg[0]) - 2
                    x1 = float(seg[1]) + 2
                    W = [w for w in line_words if w[0] >= x0]
                    W = [w for w in W if w[2] <= x1]
                    cell_words = sorted(W, key=lambda x: (x[1], x[0]))
                    row.append(cell_words)
                table.append(row)
            else:
                row = [[] for _ in range(max_len-1)]
                non_empty = False
                for seg in h[0:-1]:
                    x0 = float(seg[0]) - 2
                    x1 = float(seg[1]) + 2
                    W = [w for w in line_words if w[0] >= x0]
                    W = [w for w in W if w[2] <= x1]
                    cell_words = sorted(W, key=lambda x: (x[0]))
                    cell_start = cell_words[0][0]
                    cell_stop = cell_words[-1][2]
                    s = None
                    e = None
                    for idx, tmp in enumerate(ref_cells):
                        if cell_start <= tmp[1]:
                            s = idx
                            break
                    for idx, tmp in enumerate(ref_cells):
                        if cell_stop <= tmp[0]:
                            e = idx
                            break
                    if s is not None:
                        row[s] += cell_words
                        non_empty = True
                if non_empty:
                    table.append(row)
        return table

def make_paragraph(words, start, stop):
    block = None
    y0 = float(start) - 2
    y1 = float(y0 + (stop - start)) + 4
    W = [w for w in words if w[1] >= y0]
    W = [w for w in W if w[3] <= y1]
    W = sorted(W, key=lambda x: (x[1], x[0]))    
    if len(W) > 0:
        points = [w[0:4] for w in W]
        points = np.array(points)
        x0, y0 = np.min(points[:, 0:2], axis=0)
        x1, y1 = np.max(points[:, 2:4], axis=0)
        return [x0, y0, x1, y1, "PARA", W]
    return block

def is_y_similar(ry0, ry1, y0, y1):
    if np.abs(ry0 - y0) < 5 and np.abs(ry1 - y1) < 5:
        return True
    return False

def get_lines(cell):
    unique_y = {}
    for word in cell:
        if len(word[-1]) > 0:
            unique_y[(word[1], word[3])] = True    
    C_inds = []
    lines = []
    for y in unique_y.keys():
        ref_y0 = y[0]
        ref_y1 = y[1]
        C = []
        inds = []
        for idx, c in enumerate(cell):
            if idx not in inds:
                y0 = c[1]
                y1 = c[3]
                if is_y_similar(ref_y0, ref_y1, y0, y1):
                    C.append(c)
                    inds.append(idx)
        C = sorted(C, key=lambda x: (x[1], x[0]))
        inds = list(set(inds))
        C_inds.extend(inds)
        lines.append(C)
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
        s = set(num_lines)
        if 0 in s:
            s.remove(0)
        if len(s) == 1: 
            if 1 in s:
                new_table.append(row)
                continue
            else:                
                r = list(s)[0]                
                for x in range(r):
                    new_row = []
                    for cell in cols:
                        try:
                            l = cell[x]
                            new_row.append(l)
                        except Exception:
                            new_row.append([])
                    new_table.append(new_row)                
        else:
            max_lines_idx = np.argsort(num_lines)[-1]
            ref = cols[max_lines_idx]
            prev_rr = False
            p_rr = [[] for _ in range(len(cols))]
            rr = [[] for _ in range(len(cols))]
            for line in ref:
                y0 = min([w[1] for w in line])
                y1 = max([w[3] for w in line])
                rr[max_lines_idx].extend(line)
                for ii, col in enumerate(cols):
                    if ii == max_lines_idx: continue
                    for line2 in col:
                        yy0 = min([w[1] for w in line2])
                        yy1 = max([w[3] for w in line2])
                        if is_y_similar(y0, y1, yy0, yy1):
                            rr[ii].extend(line2)
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
    W = [x for r in table for c in r for x in c if len(x) > 0]
    W = sorted(W, key=lambda x: (x[1], x[0]))
    if len(W) > 0:
        points = [w[0:4] for w in W]
        points = np.array(points)
        x0, y0 = np.min(points[:, 0:2], axis=0)
        x1, y1 = np.max(points[:, 2:4], axis=0)
        return [x0, y0, x1, y1, "TABLE", table]
    return block

def make_blocks(segments, page_matrix, words):
    blocks = []
    for segment in segments:
        block = {}
        start, stop, label = segment
        if label == "TABLE":
            segment_matrix = page_matrix[start:stop, :]
            margins = get_segment_margins(segment_matrix)
            segment_height, segment_width = np.shape(segment_matrix)
            cropped_segment = segment_matrix[margins[0]:margins[1], :]
            table = make_cells(cropped_segment, margins, words, start, stop)
            if len(table) == 1:
                # Only 1 row, probably not a table, treat it as a paragraph
                block = make_paragraph(words, start, stop)
            else:
                block = make_table(table)
        else:
            block = make_paragraph(words, start, stop)
        if block is not None:
            blocks.append(block)    
    return blocks

