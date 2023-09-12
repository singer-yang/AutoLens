""" Aotumated lens design with curriculum learning, using RMS errors as loss function.
"""
import os
import string
import argparse
import logging
import yaml
from datetime import datetime
import torch
import deeplens
from deeplens.utils import *
from deeplens.optics import create_lens

def config():
    """ Config file for training.
    """
    # Config file
    with open('configs/auto_lens_design.yml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # Result dir
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + '-CurriculumLensDesign-' + random_string
    result_dir = f'./results/{exp_name}'
    os.makedirs(result_dir, exist_ok=True)
    args['result_dir'] = result_dir
    
    set_seed(args['seed'])
    
    # Log
    set_logger(result_dir)
    logging.info(f'EXP: {args["EXP_NAME"]}')

    # Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    logging.info(f'Using {num_gpus} GPUs')

    return args


def change_lens(lens, diag, fnum):
    """ Change lens for each curriculum step.
    """
    # sensor
    lens.r_last = diag / 2
    lens.hfov = np.arctan(lens.r_last / lens.foclen)

    # aperture
    lens.fnum = fnum
    aper_r = lens.foclen / fnum / 2
    lens.surfaces[lens.aper_idx].r = aper_r
    
    return lens


def curriculum_learning(lens, args):
    """ Curriculum learning for lens design.
    """
    lrs = [float(lr) for lr in args['lrs']]

    curriculum_steps = args['curriculum_steps']
    fnum_target = args['FNUM'] * 0.95
    fnum_start = args['FNUM_START']
    diag_target = args['DIAG'] * 1.05
    diag_start = args['DIAG_START']
    
    for step in range(args['curriculum_steps']+1):
        
        # ==> Design target for this step
        args['step'] = step
        diag1 = diag_start + (diag_target - diag_start) * np.sin(step / curriculum_steps * np.pi/2)
        fnum1 = fnum_start + (fnum_target - fnum_start) * np.sin(step / curriculum_steps * np.pi/2)
        lens = change_lens(lens, diag1, fnum1)

        lens.analysis(save_name=f'{result_dir}/step{step}_starting_point', zmx_format=True)
        lens.write_lensfile(f'{result_dir}/step{step}_starting_point.txt', write_zmx=True)
        logging.info(f'==> Curriculum learning step {step}, target: FOV {round(lens.hfov * 2 * 57.3, 2)}, DIAG {round(2 * lens.r_last, 2)}mm, F/{lens.fnum}.')
        
        # ==> Lens design using RMS errors
        iterations = 1000
        lens.refine(lrs=lrs, decay=args['ai_lr_decay'], iterations=iterations, test_per_iter=50, importance_sampling=False, result_dir=result_dir)

    # ==> Refine lens at the last step
    lens.refine(iterations=5000, test_per_iter=100, centroid=True, importance_sampling=True, result_dir=result_dir)
    logging.info('==> Training finish.')

    # ==> Final lens
    lens = change_lens(lens, args['DIAG'], args['FNUM'])


if __name__=='__main__':
    args = config()
    result_dir = args['result_dir']
    device = args['device']

    # =====> 1. Load or create lens
    if args['brute_force']:
        create_lens(rff=float(args['rff']), flange=float(args['flange']), d_aper=args['d_aper'], hfov=args['HFOV'], imgh=args['DIAG'], fnum=args['FNUM'], surfnum=args['element'], dir=result_dir)
        lens_name = f'./{result_dir}/starting_point_hfov{args["HFOV"]}_imgh{args["DIAG"]}_fnum{args["FNUM"]}.txt'
        lens = deeplens.Lensgroup(filename=lens_name)
        for i in lens.find_diff_surf():
            lens.surfaces[i].init_c()
            lens.surfaces[i].init_ai(args['ai_degree'])
            lens.surfaces[i].init_k()
            lens.surfaces[i].init_d()
    else:
        lens = deeplens.Lensgroup(filename=args['filename'])
        lens.correct_shape()
    
    lens.set_target_fov_fnum(hfov=args['HFOV'], fnum=args['FNUM'], imgh=args['DIAG'])
    logging.info(f'==> Design target: FOV {round(args["HFOV"]*2*57.3, 2)}, DIAG {args["DIAG"]}mm, F/{args["FNUM"]}, FOCLEN {round(args["DIAG"]/2/np.tan(args["HFOV"]), 2)}mm.')
    lens.analysis(save_name=f'{result_dir}/lens_starting_point')

    # =====> 2. Curriculum learning with RMS errors
    curriculum_learning(lens, args)

    # =====> 3. Analyze final result
    lens.prune(outer=0.2)
    lens.post_computation()

    logging.info(f'Actual: FOV {lens.hfov}, IMGH {lens.r_last}, F/{lens.fnum}.')
    lens.write_lensfile(f'{result_dir}/final_lens.txt', write_zmx=True)
    lens.analysis(save_name=f'{result_dir}/final_lens', zmx_format=True)
