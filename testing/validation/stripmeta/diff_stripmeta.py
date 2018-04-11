import argparse
import glob
import os

import numpy as np


np.set_printoptions(suppress=True)


class InvalidArgumentError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


def main():
    parser = argparse.ArgumentParser(description=(
        "Compares 'Strip Footprint Vertices' and "
        "'Mosaicking Alignment Statistics' sections of "
        "strip metadata text files between two directories."))

    parser.add_argument('dir1',
        help="Path to first strips directory containing *_meta.txt files.")
    parser.add_argument('dir2',
        help="Path to second strips directory containing *_meta.txt files.")

    parser.add_argument('-o', '--out', default=os.path.join(os.getcwd(), 'diff_stripmeta_results.txt'),
        help="File path of results text file (default is './diff_stripmeta_results.txt').")

    # Parse and validate arguments.
    args = parser.parse_args()
    dir1 = os.path.abspath(args.dir1)
    dir2 = os.path.abspath(args.dir2)
    outFile = os.path.abspath(args.out)
    outDir = os.path.dirname(outFile)

    if not os.path.isdir(dir1):
        parser.error("dir1 must be a directory")
    if not os.path.isdir(dir2):
        parser.error("dir2 must be a directory")
    # if os.path.isfile(outFile):
    #     parser.error("out file already exists")
    if not os.path.isdir(os.path.dirname(outDir)):
        print "Creating directory for output results file: {}".format(outDir)
        os.makedirs(outDir)

    dir1_fnames = set([os.path.basename(p) for p in glob.glob(os.path.join(dir1, '*_meta.txt'))])
    dir2_fnames = set([os.path.basename(p) for p in glob.glob(os.path.join(dir2, '*_meta.txt'))])

    fnames_comm = list(dir1_fnames.intersection(dir2_fnames))
    fnames_comm.sort()

    unifnames1 = list(dir1_fnames.difference(dir2_fnames))
    unifnames2 = list(dir2_fnames.difference(dir1_fnames))
    unifnames1.sort()
    unifnames2.sort()

    diff_fp = open(outFile, 'w')
    diff_fp.write("\n")

    if len(unifnames1) > 0 or len(unifnames2) > 0:
        diff_fp.write("*** Strip segmentation differs between dir1 and dir2 ***\n")
        if len(unifnames1) > 0:
            diff_fp.write("\nSegments unique to dir1:\n\n")
            for fname in unifnames1:
                diff_fp.write(fname + "\n")
        if len(unifnames2) > 0:
            diff_fp.write("\nSegments unique to dir2:\n\n")
            for fname in unifnames2:
                diff_fp.write(fname + "\n")
        diff_fp.write("\n")

    diff_fp.write("\n%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%\n"
                    "#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#\n\n")

    trans_diffs = []
    redundant_count1 = 0
    redundant_count2 = 0
    segs_with_scene_discreps = []

    num_jobs = len(fnames_comm)
    print("Found {} *_meta.txt files in common".format(num_jobs))
    jobnum = 0
    for stripnum, fname in enumerate(fnames_comm):
        jobnum += 1
        print "({}/{}) {}".format(jobnum, num_jobs, fname)

        metaFile1 = os.path.join(dir1, fname)
        metaFile2 = os.path.join(dir2, fname)

        meta_fp1 = open(metaFile1, 'r')
        meta_fp2 = open(metaFile2, 'r')

        skip_lines(meta_fp1, meta_fp2, 3)

        proj1, proj2 = read_lines(meta_fp1, meta_fp2, 1)

        skip_lines(meta_fp1, meta_fp2, 2)

        Xvert1, Xvert2 = read_lines(meta_fp1, meta_fp2, 1)
        Yvert1, Yvert2 = read_lines(meta_fp1, meta_fp2, 1)

        skip_lines(meta_fp1, meta_fp2, 3)

        scenes1, scenes2 = [], []
        fp, scenes = meta_fp1, scenes1
        for i in range(2):
            line = fp.readline()
            while line != '\n':
                scenes.append(line)
                line = fp.readline()
            fp, scenes = meta_fp2, scenes2

        meta_fp1.close()
        meta_fp2.close()


        diff_proj = False
        diff_vert = False
        diff_scenes = False

        if proj2 != proj1:
            diff_proj = True


        if Xvert2 != Xvert1 or Yvert2 != Yvert1:
            diff_vert = True

        coords1 = set()
        coords2 = set()

        Xvert, Yvert = Xvert1, Yvert1
        coords = coords1
        for i in range(2):
            x_coords = np.fromstring(Xvert.replace("X:", '').strip(), sep=' ')
            y_coords = np.fromstring(Yvert.replace("Y:", '').strip(), sep=' ')

            xy_coords = np.array([x_coords, y_coords]).T
            for xy in xy_coords:
                coords.add(tuple(xy))

            coords_list = list(coords)
            coords_list.sort()
            coords_array = np.array(coords_list).T.astype(np.int64)
            if Xvert == Xvert1:
                Xvert1 = "X: {} \n".format(np.array_str(coords_array[0], max_line_width=float('inf'))[1:-1])
            else:
                Xvert2 = "X: {} \n".format(np.array_str(coords_array[0], max_line_width=float('inf'))[1:-1])
            if Yvert == Yvert1:
                Yvert1 = "Y: {} \n".format(np.array_str(coords_array[1], max_line_width=float('inf'))[1:-1])
            else:
                Yvert2 = "Y: {} \n".format(np.array_str(coords_array[1], max_line_width=float('inf'))[1:-1])

            Xvert, Yvert = Xvert2, Yvert2
            coords = coords2

        num_commcords = len(coords1.intersection(coords2))
        num_unicoords1 = len(coords1.difference(coords2))
        num_unicoords2 = len(coords2.difference(coords1))


        if scenes2 != scenes1:
            diff_scenes = True

        scene_names1 = [line.split(' ')[0] for line in scenes1]
        scene_names2 = [line.split(' ')[0] for line in scenes2]

        scene_trans1 = [np.fromstring((' '.join(line.split(' ')[1:])).strip(), sep=' ') for line in scenes1]
        scene_trans2 = [np.fromstring((' '.join(line.split(' ')[1:])).strip(), sep=' ') for line in scenes2]
        scene_trans1 = np.array(scene_trans1)
        scene_trans2 = np.array(scene_trans2)

        for i in range(min(len(scene_names1), len(scene_names2))):
            if scene_names2[i] != scene_names1[i]:
                i -= 1
                break
        num_common_scenes = i + 1

        if num_common_scenes == 0:
            no_common_scenes = True
            diff_trans = np.array([[np.nan, np.nan, np.nan, np.nan]])
        else:
            no_common_scenes = False
            if num_common_scenes != max(len(scene_names1), len(scene_names2)):
                has_unique_scenes = True
            else:
                has_unique_scenes = False

            scene_trans1_comm = scene_trans1[:num_common_scenes]
            scene_trans2_comm = scene_trans2[:num_common_scenes]

            diff_trans = scene_trans2_comm - scene_trans1_comm

            redundant1 = ~np.any(scene_trans1_comm, axis=1)
            redundant2 = ~np.any(scene_trans2_comm, axis=1)
            redundant_comm = redundant1 & redundant2
            redundant1[redundant_comm] = False
            redundant2[redundant_comm] = False

            diff_trans[np.where(redundant1 | redundant2)[0], :] = np.nan

            redundant_count1 += np.count_nonzero(redundant1)
            redundant_count2 += np.count_nonzero(redundant2)

        trans_diffs.append(diff_trans)

        if no_common_scenes or has_unique_scenes:
            segs_with_scene_discreps.append("{}, {}\n".format(stripnum+1, fname))


        diff_fp.write("\n\n{}, strip segment name = {}\n\n".format(stripnum+1, fname))

        dir = dir1
        proj = proj1
        Xvert = Xvert1
        Yvert = Yvert1
        num_unicoords = num_unicoords1
        scenes = scenes1
        for i in range(2):
            diff_fp.write("\n{}:\n\n".format(dir))

            if diff_proj:
                diff_fp.write("{}\n".format(proj))

            diff_fp.write(Xvert)
            diff_fp.write(Yvert)
            diff_fp.write("Num common verts: {}\n".format(num_commcords))
            if diff_vert:
                diff_fp.write("Num unique verts: {}\n".format(num_unicoords))
            else:
                diff_fp.write("::VERTS MATCH::\n")

            diff_fp.write("\n{}\n".format(''.join(scenes)))

            dir = dir2
            proj = proj2
            Xvert = Xvert2
            Yvert = Yvert2
            num_unicoords = num_unicoords2
            scenes = scenes2

            if i == 0:
                diff_fp.write("---------------------------------------------------------\n")

        diff_fp.write("\n")
        if diff_scenes:
            if no_common_scenes:
                diff_fp.write(">>> No common scenes for trans diff <<<\n")
            else:
                diff_fp.write("{}\n".format(np.array_str(diff_trans)))
                if has_unique_scenes:
                    diff_fp.write("--> Only {} scenes in common for trans diff <--\n".format(num_common_scenes))
        else:
            diff_fp.write("[[TRANS MATCH]]\n")

        diff_fp.write("\n\n%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%\n"
                          "#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#%#\n")

    diff_fp.write("\n\n\n")

    # trans_diffs_extrema = np.vstack([td[np.nanargmax(np.abs(td), axis=0), np.arange(td.shape[1])] for td in trans_diffs])
    trans_diffs_extrema = []
    for td in trans_diffs:
        if np.any(~np.isnan(td)):
            trans_diffs_extrema.append(td[np.nanargmax(np.abs(td), axis=0), np.arange(td.shape[1])])
        else:
            trans_diffs_extrema.append(np.array([np.nan, np.nan, np.nan, np.nan]))
    trans_diffs_extrema = np.vstack(trans_diffs_extrema)

    col_labels = "RMSE, dz, dx, dy"

    line_format = '{:<'+str(len(str(trans_diffs_extrema.shape[0])))+'} {}'
    diff_fp.write("Per-segment AbsMax difference (dir2 - dir1):\n\n{}\n{}\n\n".format(
        line_format.format("#", " "+np.array2string(np.array([s.strip() for s in col_labels.split(',')]))),
        '\n'.join([line_format.format(n+1, line) for n, line in enumerate(np.array2string(trans_diffs_extrema, max_line_width=np.inf, threshold=np.inf).split('\n'))])
    ))

    stats_array = trans_diffs_extrema

    diff_fp.write("\n".join([
        "Mean   : "+np.array2string(np.nanmean(stats_array, axis=0), max_line_width=np.inf, threshold=np.inf),
        "Median : "+np.array2string(np.nanmedian(stats_array, axis=0), max_line_width=np.inf, threshold=np.inf),
        "StdDev : "+np.array2string(np.nanstd(stats_array, axis=0), max_line_width=np.inf, threshold=np.inf),
        "AbsMax : "+np.array2string(stats_array[np.nanargmax(np.abs(stats_array), axis=0), np.arange(stats_array.shape[1])], max_line_width=np.inf, threshold=np.inf),
        "AbsMax#: "+np.array2string(1+np.nanargmax(np.abs(stats_array), axis=0), max_line_width=np.inf, threshold=np.inf)
        ]
    ))

    diff_fp.write("\n\n")

    diff_fp.write("\nNum redundant scenes unique to dir1: {}".format(redundant_count1))
    diff_fp.write("\nNum redundant scenes unique to dir2: {}".format(redundant_count2))

    diff_fp.write("\n")

    if len(segs_with_scene_discreps) > 0:
        diff_fp.write("\n\n*** The following segments have discrepancies in scene content ***\n\n")
        for seg in segs_with_scene_discreps:
            diff_fp.write(seg)

    diff_fp.close()

    np.savetxt(outFile+'_AbsMaxDiff.csv', stats_array, delimiter=',', fmt='%0.3f')

    print "Done!"


def read_lines(fp1, fp2, num=0):
    str1, str2 = "", ""
    for i in range(num):
        str1 += fp1.readline()
        str2 += fp2.readline()
    return str1, str2


def skip_lines(fp1, fp2, num=0):
    for i in range(num):
        fp1.readline()
        fp2.readline()



if __name__ == '__main__':
    main()
