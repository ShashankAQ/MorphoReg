"""Audit native images, then copy reviewed kept/debris outputs without moving originals."""
import argparse
import csv
import shutil
from pathlib import Path

def border_values(a, width=8):
    h,w = a.shape
    bw = max(1, min(width, h//4, w//4))
    return np.concatenate([a[:bw,:].ravel(), a[-bw:,:].ravel(), a[:,:bw].ravel(), a[:,-bw:].ravel()])

def heuristic_cell_mask(raw, halo_dilate=2):
    x = raw.astype(np.float32)
    b = border_values(x, 8)
    bg = float(np.median(b))
    mad = float(np.median(np.abs(b-bg)))
    sigma = 1.4826*mad
    thr = max(2.0, 4.0*sigma)
    m = np.abs(x-bg) > thr
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    lab,n = ndi.label(m)
    if n > 0:
        sizes = ndi.sum(m, lab, index=np.arange(1,n+1))
        m = lab == (int(np.argmax(sizes))+1)
    if halo_dilate > 0 and np.any(m):
        m = ndi.binary_dilation(m, iterations=halo_dilate)
    return m.astype(bool), bg, mad, sigma, thr

def touches_border(mask):
    return bool(mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any())

def measurements(raw, halo_dilate=2):
    mask,bg,mad,sigma,threshold=heuristic_cell_mask(raw,halo_dilate)
    ys,xs=np.where(mask)
    width=int(xs.max()-xs.min()+1) if len(xs) else 0
    height=int(ys.max()-ys.min()+1) if len(ys) else 0
    return {'mask_area_fraction':float(mask.mean()),'bbox_width_px':width,
            'bbox_height_px':height,'aspect_ratio':max(width,height)/max(1,min(width,height)),
            'bbox_max_dim':max(width,height),'touches_border':touches_border(mask),
            'background_median':bg,'border_mad':mad,'threshold':threshold}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,help='segmentation_manifest.csv for audit')
    parser.add_argument('--review-csv',required=True,type=Path,help='Audit CSV to create, or reviewed CSV to apply')
    parser.add_argument('--apply',action='store_true',help='Copy files according to manually entered keep/debris decisions')
    parser.add_argument('--output',type=Path,help='New/empty folder for kept/debris copies')
    parser.add_argument('--area-threshold',type=float,default=.20)
    parser.add_argument('--aspect-threshold',type=float,default=2.0)
    parser.add_argument('--extent-threshold',type=int,default=100)
    parser.add_argument('--halo-dilate',type=int,default=2)
    args=parser.parse_args()
    if args.apply:
        if not args.review_csv.is_file() or args.output is None:
            parser.error('--apply needs an existing reviewed CSV and --output')
        out=args.output.resolve()
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            parser.error('Output must be new or empty')
        with args.review_csv.open(newline='',encoding='utf-8') as handle:
            rows=list(csv.DictReader(handle))
        if not rows:parser.error('Review CSV is empty')
        plan=[]
        for index,row in enumerate(rows):
            if row.get('status')!='ok':continue
            decision=row.get('decision','').strip().lower()
            if decision not in {'keep','debris'}:
                parser.error('Every successfully audited row needs decision keep or debris before --apply')
            for key in ['native_image','native_mask','zoomed_image','zoomed_mask']:
                src=Path(row[key]).resolve()
                if not src.is_file():parser.error(f'Missing file: {src}')
                if out==src.parent or out in src.parents or src.parent in out.parents:
                    parser.error('Filtered output must be outside the source image trees')
                # Row-specific folders avoid collisions and retain image/mask association.
                dst=out/decision/f'{index:07d}'/key/src.name
                plan.append((src,dst,decision))
        if not plan:parser.error('No successfully audited files to copy')
        out.mkdir(parents=True)
        with (out/'filter_manifest.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=['source','destination','decision']);writer.writeheader()
            for src,dst,decision in plan:
                dst.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,dst)
                writer.writerow({'source':str(src),'destination':str(dst),'decision':decision})
                handle.flush()
        print(f'Copied {len(plan)} image/mask files; originals unchanged: {out}')
        return 0
    if not args.manifest or not args.manifest.is_file():parser.error('Audit requires --manifest')
    if args.review_csv.exists():parser.error('Review CSV already exists; choose a new filename')
    if not 0<args.area_threshold<=1 or args.aspect_threshold<=0 or args.extent_threshold<1 or args.halo_dilate<0:
        parser.error('Invalid audit thresholds')
    global np,ndi
    import numpy as np
    from scipy import ndimage as ndi
    import tifffile as tiff
    with args.manifest.open(newline='',encoding='utf-8') as handle:
        rows=list(csv.DictReader(handle))
    if not rows:parser.error('Segmentation manifest is empty')
    results=[]
    for row in rows:
        if row['status']!='ok':continue
        result={**row,'decision':'','flags':''}
        try:
            raw=tiff.imread(row['native_image'])
            if raw.ndim!=2:raise ValueError('Expected single-frame grayscale image')
            stats=measurements(raw,args.halo_dilate)
            reasons=[]
            if stats['mask_area_fraction']>=args.area_threshold:reasons.append('large_area')
            if .05<=stats['mask_area_fraction']<.40 and stats['aspect_ratio']>=args.aspect_threshold:reasons.append('elongated')
            if stats['bbox_max_dim']>=args.extent_threshold and stats['mask_area_fraction']<.40:reasons.append('large_extent')
            if stats['mask_area_fraction']==0:reasons.append('empty_heuristic_mask')
            result.update(stats,flags=';'.join(reasons),status='ok')
        except Exception as exc:
            result.update(status='audit_failed',error=str(exc))
        results.append(result)
    if not results:parser.error('No successfully segmented objects in manifest')
    args.review_csv.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in results for key in row))
    with args.review_csv.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(results)
    print(f'Wrote {len(results)} rows to {args.review_csv}. Review flags and enter keep/debris in decision.')
    return 1 if any(row['status']=='audit_failed' for row in results) else 0


if __name__=='__main__':
    raise SystemExit(main())
