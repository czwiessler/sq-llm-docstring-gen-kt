#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 1 August 2018 by chen.wu@icrar.org

import os
import os.path as osp
import math
import warnings
import csv
from collections import defaultdict
import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy.coordinates import SkyCoord

from string import Template
from collections import defaultdict
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


anno_tpl_str = """<annotation>
        <folder>${one_level_up_folder}</folder>
        <filename>${file_id}.png</filename>
        <source>
                <database>The EMU Database</database>
                <annotation>EMU2018</annotation>
                <image>gama23-emu</image>
                <flickrid>${file_id}</flickrid>
        </source>
        <owner>
                <flickrid>emuid</flickrid>
                <name>emu-member</name>
        </owner>
        <size>
                <width>${pic_size_width}</width>
                <height>${pic_size_height}</height>
                <depth>3</depth>
        </size>
        <segmented>0</segmented>
        ${bbox}
</annotation>
"""

bbox_tpl_str = """
        <object>
                <name>${class_name}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>${xmin}</xmin>
                        <ymin>${ymin}</ymin>
                        <xmax>${xmax}</xmax>
                        <ymax>${ymax}</ymax>
                </bndbox>
        </object>"""

anno_tpl = Template(anno_tpl_str)
bbox_tpl = Template(bbox_tpl_str)


# key - fits path, value -  a list of boxes, each denoting a complete source

def _gen_single_bbox(fits_fn, ra, dec, major, minor, pa, major_scale=1.2, png_size=None, padx=True):
    """
    Form the bbox BEFORE converting wcs to the pixel coordinates
    major and mior are in arcsec
    """
    ra = float(ra)
    dec = float(dec)
    hdulist = pyfits.open(fits_fn)
    w = pywcs.WCS(hdulist[0].header)
    x_height, y_width = hdulist[0].data.shape
    ang = major * major_scale / 3600.0 
    res_x = abs(hdulist[0].header['CDELT1'])
    width = int(ang / res_x)
    #print("width = {}".format(width))

    xmin = ra + ang #actually semi-major
    ymin = dec - ang
    xp_min, yp_min = w.wcs_world2pix([[xmin, ymin, 0, 0]], 0)[0][0:2]
    #print(xp_min, yp_min)
    xp_min = round(xp_min)
    yp_min = round(yp_min)
    xmax = ra - ang
    ymax = dec + ang
    xp_max, yp_max = w.wcs_world2pix([[xmax, ymax, 0, 0]], 0)[0][0:2]
    xp_max = round(xp_max)
    yp_max = round(yp_max)
    #print('x', xmin, xmax, xp_min, xp_max, xp_max - xp_min)
    #print('y', ymin, ymax, yp_min, yp_max, yp_max - yp_min)

    # Astronomy pixel (0,0) starts from bottom left, but computer vision images
    # (PNG, JPEG) starts from top left, so need to convert them again
    t = yp_min
    yp_min = y_width - yp_max
    yp_max = y_width - t

    # crop it around the border
    xp_min = int(math.ceil(max(xp_min, 1)))
    yp_min = int(math.ceil(max(yp_min, 1)))
    xp_max = int(math.floor(min(xp_max, x_height - 1)))
    yp_max = int(math.floor(min(yp_max, y_width - 1)))

    if (padx and (xp_max - xp_min < width)):
        dw = width - (xp_max - xp_min)
        xp_max += dw / 2
        xp_max = int(math.floor(min(xp_max, x_height - 1)))
        xp_min -= dw / 2
        xp_min = int(math.ceil(max(xp_min, 1)))
    #print('x', xmin, xmax, xp_min, xp_max, xp_max - xp_min)
    #print('y', ymin, ymax, yp_min, yp_max, yp_max - yp_min)
    # if ((png_size is not None) and (png_size != origin_pic_size)):  # need to scale the bbox
    #     ratio = float(png_size) / origin_pic_size
    #     xp_min = int(ratio * xp_min)
    #     yp_min = int(ratio * yp_min)
    #     xp_max = int(ratio * xp_max)
    #     yp_max = int(ratio * yp_max)
    return (xp_min, yp_min, xp_max, yp_max, x_height, y_width)

def convert_box2sky(detpath, fitsdir, outpath, threshold=0.8):
    """
    Convert output of ClaRAN (boxes) into sky coordinates
    """
    with open(detpath, 'r') as fin:
        mylist = fin.read().splitlines()
        for line in mylist:
            ll = line.split()
            # print(ll)
            if (float(ll[1]) < threshold):
                continue
            x1, y1_, x2, y2_ = [float(x) for x in ll[2:]]
            fname = osp.join(fitsdir, ll[0].split('_')[0] + '.fits')
            # print(fname)
            file = pyfits.open(fname)
            height, _ = file[0].data.shape
            fhead = file[0].header
            y2 = height - y1_
            y1 = height - y2_
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            warnings.simplefilter("ignore")
            w = pywcs.WCS(fhead, naxis=2)
            warnings.simplefilter("default")
            ra, dec = w.wcs_pix2world([[cx, cy]], 0)[0]
            print('%.4f,%.4f,%s' % (ra, dec, osp.basename(fname)))


def _convert_source2box(source, fits_dir, table_name, conn, known_fits=[]):
    """
    for each component C in the catalog, this function outputs two things:

        1. The name of the split file S to which, component C belongs
        2. Within S, the location of C in pixel coordinates (x1, y1, x2, y2)

    Note that the pixel cooridnates start from top left (rather than bottom left)
    This is used to compared the ClaRAN box

    source:   a list of components inside a source
    """
    # if (len(source) > 1):
    #     print(source)
    pix_resolution = 0.0011 #pixel angular resolution
    agg_res = np.zeros([len(source), 4])
    #thirtyarcmin_960mhz
    for i, c in enumerate(source):
        ra, dec, major, minor, pa = float(c[6]), float(c[8]), float(c[14]), \
                                    float(c[16]), float(c[18])
        sqlStr = "select fileid from %s where coverage ~ scircle " % table_name +\
                 "'<(%fd, %fd), %fd>'" % (ra, dec, pix_resolution)
        cur = conn.cursor(sqlStr)
        cur = conn.cursor()
        cur.execute(sqlStr)
        res = cur.fetchall()
        if (not res or len(res) == 0):
            print("fail to find fits file {0}".format(sqlStr))
            return None
        fits_path = None
        for j in range(len(res)):
            apath = osp.join(fits_dir, res[j][0])
            if (apath in known_fits):
                continue
            else:
                fits_path = apath
                break
        if (fits_path is None):
            return None
        if (not (osp.exists(fits_path))):
            raise Exception('fits file not found %s' % fits_path)
        box_re = _gen_single_bbox(fits_path, ra, dec, major, minor, pa)
        agg_res[i, :] = box_re[0:4]
    x1 = int(np.min(agg_res[:, 0]))
    y1 = int(np.min(agg_res[:, 1]))
    x2 = int(np.max(agg_res[:, 2]))
    y2 = int(np.max(agg_res[:, 3]))
    return (fits_path, x1, y1, x2, y2, box_re[4], box_re[5], 
            '%dC' % len(source))


def _get_fits_mbr(fin, row_ignore_factor=10):
    hdulist = pyfits.open(fin)
    data = hdulist[0].data
    wcs = pywcs.WCS(hdulist[0].header)
    width = data.shape[1]
    height = data.shape[0]

    bottom_left = [0, 0, 0, 0]
    top_left = [0, height - 1, 0, 0]
    top_right = [width - 1, height - 1, 0, 0]
    bottom_right = [width - 1, 0, 0, 0]

    def pix2sky(pix_coord):
        return wcs.wcs_pix2world([pix_coord], 0)[0][0:2]

    ret = np.zeros([4, 2])
    ret[0, :] = pix2sky(bottom_left)
    ret[1, :] = pix2sky(top_left)
    ret[2, :] = pix2sky(top_right)
    ret[3, :] = pix2sky(bottom_right)
    RA_min, DEC_min, RA_max, DEC_max = np.min(ret[:, 0]),   np.min(ret[:, 1]),  np.max(ret[:, 0]),  np.max(ret[:, 1])
    
    # http://pgsphere.projects.pgfoundry.org/types.html
    sqlStr = "SELECT sbox '((%10fd, %10fd), (%10fd, %10fd))'" % (RA_min, DEC_min, RA_max, DEC_max)
    return sqlStr

def _setup_db_pool():
    from psycopg2.pool import ThreadedConnectionPool
    return ThreadedConnectionPool(1, 3, database='chen', user='chen')

def convert_sky2box(catalog_csv_file, split_fits_dir, table_name):
    """
    1. work out which fits file each record belongs to
    """
    # build out the fits header cache to handle queries like:
    # does this point inside this fits file?
    fits_box_dict = defaultdict(list)
    with open(catalog_csv_file, 'r') as fin:
        cpnlist = fin.read().splitlines()
    cpnlist = sorted(cpnlist[1:], key=lambda x: int(x.split(',')[0]))
    last_sid = cpnlist[0].split(',')[0]
    last_source = []
    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()
    for idx, cpnline in enumerate(cpnlist):
        cpn = cpnline.split(',')
        sid = cpn[0]
        if (last_sid != sid):
            known_fits_fid = []
            while (1):
                ret = _convert_source2box(last_source, split_fits_dir, 
                                          table_name, conn, known_fits=known_fits_fid)
                if (ret is None):
                    break
                fits_box_dict[ret[0]].append(ret[1:])
                known_fits_fid.append(ret[0])
                
            last_source = []
        last_source.append(cpn)
        last_sid = sid
        if (idx % 100 == 0):
            print('Processed %d components' % (idx + 1))
    g_db_pool.putconn(conn)
    return fits_box_dict

def write_annotations(fits_box_dict, out_dir):
    for fits_path, boxes in fits_box_dict.items():
        fp, ext = osp.splitext(fits_path)
        upper_dir = fp.split(os.sep)[-1]
        file_id = osp.basename(fits_path)
        radio_sources = []
        for box in boxes:
            obj_dict = {'class_name': box[-1], 'xmin': box[0], 'ymin': box[1],
                    'xmax': box[2], 'ymax': box[3]}
            obj_str = bbox_tpl.safe_substitute(obj_dict)
            radio_sources.append(obj_str)
        anno_str = anno_tpl.\
        safe_substitute({'file_id': upper_dir, 'bbox': ''.join(radio_sources),
                         'pic_size_height': box[4], 'pic_size_width': box[5], 
                         'one_level_up_folder': upper_dir})
        anno_fn = osp.join(out_dir, file_id.replace(ext, '.xml'))
        with open(anno_fn, 'w') as fo:
            fo.write(anno_str)

def _draw_boxes(png_file, boxes, out_dir, box_color='white'):
    im = cv2.imread(png_file)
    h, w, _ = im.shape
    my_dpi = 100.0
    fig = plt.figure()
    fig.set_size_inches(h / my_dpi, w / my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_xlim([0, w])
    ax.set_ylim([h, 0])
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')

    for i in range(len(boxes)):
        bbox = boxes[i, :]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=box_color, linewidth=1.0)
            )
    
    plt.axis('off')
    plt.draw()
    plt.savefig(osp.join(out_dir, osp.basename(png_file).replace('.png', '_b.png')))

def check_annotations(anno_dir, png_dir, out_dir):
    """
    check if the annotation looks right by drawing boxes on the png
    and dump the result to the out_dir 
    """
    import xml.etree.ElementTree as ET
    for fn in os.listdir(anno_dir):
        filename = osp.join(anno_dir, fn)
        if (not filename.endswith('_E1.xml')):
            continue
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes[ix, :] = [x1, y1, x2, y2]
            #cls = obj.find('name').text.lower().strip()
        png_file = osp.join(png_dir, fn.replace('.xml', '.png'))
        _draw_boxes(png_file, boxes, out_dir)
        #break

def check_predictions(pred_dir, png_dir, out_dir, threshold=0.1):
    """
    """
    pred_dict = defaultdict(list)
    for fn in os.listdir(pred_dir):
        filename = osp.join(pred_dir, fn)
        if (not filename.endswith('c.txt')):
            continue
        with open(filename, 'r') as fin:
            mylist = fin.read().splitlines()
            for line in mylist:
                fds = line.split()
                score = fds[1]
                emu_id = fds[0]
                if (float(score) < threshold):
                    continue
                pred_dict[emu_id].append(fds[2:])

    for fn in os.listdir(png_dir):
        emu_id, _ = osp.splitext(fn.replace('_b', ''))
        if (emu_id in pred_dict):
            objs = pred_dict[emu_id]
            num_objs = len(objs)
            print(emu_id, num_objs)
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            for ix, bbox in enumerate(objs):
                x1 = float(bbox[0])
                y1 = float(bbox[1])
                x2 = float(bbox[2])
                y2 = float(bbox[3])
                boxes[ix, :] = [x1, y1, x2, y2]
            png_file = osp.join(png_dir, fn)
            _draw_boxes(png_file, boxes, out_dir, box_color='red')

def build_fits_cutout_index(fits_cutout_dir,
                            prefix='gama_low_all_corrected_clipped',
                            tablename='onedegree'):
    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()
    for fn in os.listdir(fits_cutout_dir):
        if (not fn.startswith(prefix)):
            continue
        if (not fn.endswith('.fits')):
            continue
        if (fn.find('-') < 0):
            continue
        fits_fn = osp.join(fits_cutout_dir, fn)
        sqlStr = _get_fits_mbr(fits_fn)
        cur = conn.cursor()
        cur.execute(sqlStr)
        res = cur.fetchall()
        if (not res or len(res) == 0):
            errMsg = "fail to calculate sbox {0}".format(sqlStr)
            print(errMsg)
            raise Exception(errMsg)
        coverage = res[0][0]
        sqlStr = """INSERT INTO {0}(coverage,fileid) VALUES('{1}','{2}')"""
        sqlStr = sqlStr.format(tablename, coverage, fn)
        print(sqlStr)
        cur.execute(sqlStr)
        conn.commit()
    g_db_pool.putconn(conn)

def change_file_names(fits_dir, png_dir):
    """
    rename the image file names (both fits and png) to something similar to
    RGZ, e.g.
    EMUJ144637.3+591919_960MHz_1deg.fits
    EMUJ144637.3+591919_1368MHz_30arcmin.png
    also change the name in the postgresql database
    """

    def round_second(t, decimal=3):
        if (t.endswith('s')):
            t = t[:-1]
        l, r = t.split('.')
        if (len(r) < decimal):
            r += '0' * (decimal - len(r))
        return l + '.' + r[0:decimal]

    def clean_pos(c):
        l, r = c.to_string('hmsdms').split()
        pos = round_second(l) + ' ' + round_second(r)
        for r in 'hmsd ':
            pos = pos.replace(r, '')
        return 'EMUJ%s' % pos
    
    def get_table_name():
        t = '_'.join(png_dir.split(os.sep)[-1].split('_')[2:])
        return t.replace('1deg', 'onedegree').replace('30arcmin', 'thirtyarcmin').\
                 replace('MHz', 'mhz')
        
    fits_suffix = fits_dir.split(os.sep)[-1].split('split_fits_')[0]
    png_suffix = png_dir.split(os.sep)[-1].split('split_png_')[0]
    if (fits_suffix != png_suffix):
        raise Exception('%s != %s' % (fits_suffix, png_suffix))
    
    tbl = get_table_name()
    
    tbl_type_dict = dict()
    tbl_type_dict['thirtyarcmin_960mhz'] = 'E1'
    tbl_type_dict['thirtyarcmin_1368mhz'] = 'E2'
    tbl_type_dict['onedegree_960mhz'] = 'E3'
    tbl_type_dict['onedegree_1368mhz'] = 'E4'
    dtype = tbl_type_dict[tbl]

    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()
    for fn in os.listdir(fits_dir):
        basename, ext = osp.splitext(fn)
        if (not ext in ['.fits', '.FITS']):
            continue
        png_path = osp.join(png_dir, basename  + '.png')
        if (not osp.exists(png_path)):
            raise Exception('PNG file not found %s' % png_path)

        fits_path = osp.join(fits_dir, fn)
        # hdulist = pyfits.open(fits_path)
        # data = hdulist[0].data
        # wcs = pywcs.WCS(hdulist[0].header)
        # width = data.shape[1]
        # height = data.shape[0]
        # pix_coord = [width // 2, height // 2, 0, 0]
        # center_sky = wcs.wcs_pix2world([pix_coord], 0)[0][0:2]
        # c = SkyCoord(center_sky[0], center_sky[1], frame='fk5', unit='deg')
        # new_nm = clean_pos(c) + '.fits'
        new_nm = basename + '_%s' % dtype + '.fits'
        os.rename(fits_path, osp.join(fits_dir, new_nm))
        os.rename(png_path, osp.join(png_dir, new_nm.replace('.fits', '.png')))
        sqlStr = """UPDATE %s SET fileid = '%s' WHERE fileid = '%s' """
        sqlStr = sqlStr % (tbl, new_nm, fn)
        print(sqlStr)
        cur = conn.cursor()
        cur.execute(sqlStr)
        conn.commit()

    print('table: %s' % tbl)
    g_db_pool.putconn(conn)

def regen_E1_annotations(emu_path):
    fits_fn_path = osp.join(emu_path, 'split_fits_30arcmin_960MHz')
    catalog_csv = osp.join(emu_path, '960SglCtrDblRevTpl.csv')
    print("Doing fits_box_dict")
    fits_box_dict = convert_sky2box(catalog_csv, fits_fn_path, 'thirtyarcmin_960mhz')
    print("Done fits_box_dict")
    write_annotations(fits_box_dict, osp.join(emu_path, 'annotations'))
        
if __name__ == '__main__':
    """ detpath = "/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/results"\
    "/RGZ2017/pleiades/comp4_det_testD4_2_3.txt"

    fitsdir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/RGZdevkit2017/RGZ2017/FITSImages'
    convert_box2sky(detpath, fitsdir, '/tmp') """
    emu_path = '/Users/chen/gitrepos/ml/rgz_rcnn/data/EMU_GAMA23'
    regen_E1_annotations(emu_path)
    # fits_fn = emu_path + '/split_fits/' + \
    #           '1deg/gama_linmos_corrected_clipped4-0.fits'
    # #fits_fn_path = osp.join(emu_path, 'split_fits_30arcmin_960MHz')
    # #build_fits_cutout_index(fits_fn_path, tablename='onedegree_1368mhz', prefix='gama_linmos_corrected_clipped')
    # #catalog_csv = osp.join(emu_path, '1368SglCtrDblRevTpl.csv')
    # #catalog_csv = osp.join(emu_path, '960SglCtrDblRevTpl.csv')
    # #fits_box_dict = convert_sky2box(catalog_csv, fits_fn_path, 'thirtyarcmin_960mhz')
    # #write_annotations(fits_box_dict, osp.join(emu_path, 'annotations'))
    # #change_file_names(fits_fn_path, osp.join(emu_path, 'split_png_1deg_1368MHz'))
    # anno_dir = osp.join(emu_path, 'emu_claran_dataset', 'annotations')
    # png_dir = osp.join(emu_path, 'emu_claran_dataset', 'png')
    # out_dir = osp.join(emu_path, 'emu_claran_dataset', 'check_anno')
    # pred_dir = osp.join(emu_path, 'emu_claran_dataset', 'predictions', '600_w_3088')
    # #check_annotations(anno_dir, png_dir, out_dir)
    # png_dir = out_dir
    # out_dir = osp.join(emu_path, 'emu_claran_dataset', 'check_pred', '600_w_3088')
    # check_predictions(pred_dir, png_dir, out_dir)