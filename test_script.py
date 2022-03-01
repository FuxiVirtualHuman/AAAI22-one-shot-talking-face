import os
import numpy as np
import torch
import yaml
from models.generator import OcclusionAwareGenerator
from models.keypoint_detector import KPDetector
import argparse
import imageio
from models.util import draw_annotation_box
from models.transformer import Audio2kpTransformer
from scipy.io import wavfile
from tools.interface import read_img,get_img_pose,get_pose_from_audio,get_audio_feature_from_audio,\
    parse_phoneme_file,load_ckpt
import config

def normalize_kp(kp_source, kp_driving, kp_driving_initial,
                 use_relative_movement=True, use_relative_jacobian=True):

    kp_new = {k: v for k, v in kp_driving.items()}
    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        # kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def test_with_input_audio_and_image(img_path, audio_path,phs, generator_ckpt, audio2pose_ckpt, save_dir="samples/results"):
    with open("config_file/vox-256.yaml") as f:
        config = yaml.load(f)
    # temp_audio = audio_path
    # print(audio_path)
    cur_path = os.getcwd()

    sr,_ = wavfile.read(audio_path)
    if sr!=16000:
        temp_audio = os.path.join(cur_path,"samples","temp.wav")
        command = "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, temp_audio)
        os.system(command)
    else:
        temp_audio = audio_path


    opt = argparse.Namespace(**yaml.load(open("config_file/audio2kp.yaml")))

    img = read_img(img_path).cuda()

    first_pose = get_img_pose(img_path)#.cuda()

    audio_feature = get_audio_feature_from_audio(temp_audio)
    frames = len(audio_feature) // 4
    frames = min(frames,len(phs["phone_list"]))

    tp = np.zeros([256, 256], dtype=np.float32)
    draw_annotation_box(tp, first_pose[:3], first_pose[3:])
    tp = torch.from_numpy(tp).unsqueeze(0).unsqueeze(0).cuda()
    ref_pose = get_pose_from_audio(tp, audio_feature, audio2pose_ckpt)
    torch.cuda.empty_cache()
    trans_seq = ref_pose[:, 3:]
    rot_seq = ref_pose[:, :3]



    audio_seq = audio_feature#[40:]
    ph_seq = phs["phone_list"]


    ph_frames = []
    audio_frames = []
    pose_frames = []
    name_len = frames

    pad = np.zeros((4, audio_seq.shape[1]), dtype=np.float32)

    for rid in range(0, frames):
        ph = []
        audio = []
        pose = []
        for i in range(rid - opt.num_w, rid + opt.num_w + 1):
            if i < 0:
                rot = rot_seq[0]
                trans = trans_seq[0]
                ph.append(31)
                audio.append(pad)
            elif i >= name_len:
                ph.append(31)
                rot = rot_seq[name_len - 1]
                trans = trans_seq[name_len - 1]
                audio.append(pad)
            else:
                ph.append(ph_seq[i])
                rot = rot_seq[i]
                trans = trans_seq[i]
                audio.append(audio_seq[i * 4:i * 4 + 4])
            tmp_pose = np.zeros([256, 256])
            draw_annotation_box(tmp_pose, np.array(rot), np.array(trans))
            pose.append(tmp_pose)

        ph_frames.append(ph)
        audio_frames.append(audio)
        pose_frames.append(pose)

    audio_f = torch.from_numpy(np.array(audio_frames,dtype=np.float32)).unsqueeze(0)
    poses = torch.from_numpy(np.array(pose_frames, dtype=np.float32)).unsqueeze(0)
    ph_frames = torch.from_numpy(np.array(ph_frames)).unsqueeze(0)
    bs = audio_f.shape[1]
    predictions_gen = []

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()

    ph2kp = Audio2kpTransformer(opt).cuda()

    load_ckpt(generator_ckpt, kp_detector=kp_detector, generator=generator,ph2kp=ph2kp)


    ph2kp.eval()
    generator.eval()
    kp_detector.eval()

    with torch.no_grad():
        for frame_idx in range(bs):
            t = {}

            t["audio"] = audio_f[:, frame_idx].cuda()
            t["pose"] = poses[:, frame_idx].cuda()
            t["ph"] = ph_frames[:,frame_idx].cuda()
            t["id_img"] = img

            kp_gen_source = kp_detector(img, True)

            gen_kp = ph2kp(t,kp_gen_source)
            if frame_idx == 0:
                drive_first = gen_kp

            norm = normalize_kp(kp_source=kp_gen_source, kp_driving=gen_kp, kp_driving_initial=drive_first)
            out_gen = generator(img, kp_source=kp_gen_source, kp_driving=norm)

            predictions_gen.append(
                (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))


    log_dir = save_dir
    os.makedirs(os.path.join(log_dir, "temp"),exist_ok=True)

    f_name = os.path.basename(img_path)[:-4] + "_" + os.path.basename(audio_path)[:-4] + ".mp4"
    # kwargs = {'duration': 1. / 25.0}
    video_path = os.path.join(log_dir, "temp", f_name)
    print("save video to: ", video_path)
    imageio.mimsave(video_path, predictions_gen, fps=25.0)

    # audio_path = os.path.join(audio_dir, x['name'][0].replace(".mp4", ".wav"))
    save_video = os.path.join(log_dir, f_name)
    cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
    os.system(cmd)
    os.remove(video_path)






if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--img_path", type=str, default=None, help="path of the input image ( .jpg ), preprocessed by image_preprocess.py")
    argparser.add_argument("--audio_path", type=str, default=None, help="path of the input audio ( .wav )")
    argparser.add_argument("--phoneme_path", type=str, default=None, help="path of the input phoneme. It should be note that the phoneme must be consistent with the input audio")
    argparser.add_argument("--save_dir", type=str, default="samples/results", help="path of the output video")
    args = argparser.parse_args()

    phoneme = parse_phoneme_file(args.phoneme_path)
    test_with_input_audio_and_image(args.img_path,args.audio_path,phoneme,config.GENERATOR_CKPT,config.AUDIO2POSE_CKPT,args.save_dir)
