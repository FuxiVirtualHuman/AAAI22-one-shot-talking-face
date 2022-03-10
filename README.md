# One-shot Talking Face Generation from Single-speaker Audio-Visual Correlation Learning (AAAI 2022)

#### [Paper](https://arxiv.org/pdf/2112.02749.pdf) | [Demo](https://www.youtube.com/watch?v=HHj-XCXXePY)

#### Requirements

- Python >= 3.6 , Pytorch >= 1.8 and ffmpeg
- Set up [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace)
  - We use the OpenFace tools to extract the initial pose of the reference image
  - Make sure you have installed this tool, and set the `OPENFACE_POSE_EXTRACTOR_PATH` in `config.py`. For example, it should be the absolute path of  the "`FeatureExtraction.exe`" for Windows.
- Other requirements are listed in the 'requirements.txt'



#### Pretrained Checkpoint

Please download the pretrained checkpoint from [google-drive](https://drive.google.com/file/d/1mjFEozPR_2vMaVRMd9Agk_sU1VaiUYMl/view?usp=sharing) and unzip it to the directory (`/checkpoints`). Or manually modify the settings of `GENERATOR_CKPT` and `AUDIO2POSE_CKPT` in the `config.py`.



#### Extract phoneme

We employ the [CMU phoneset](https://github.com/cmusphinx/cmudict) to represent phonemes, the extra 'SIL' means silence. All the phonesets can be seen in '`phindex.json`'.

We have extracted the phonemes for the audios in the '`sample/audio`'  directory. For other audios, you can extract the phonemes by other ASR tools and then map them to the CMU phoneset. Or email to wangsuzhen@corp.netease.com  for help.



#### Generate Demo Results

```
python test_script.py --img_path xxx.jpg --audio_path xxx.wav --phoneme_path xxx.json --save_dir "YOUR_DIR"
```

Note that the input images must keep the same height and width and the face should be appropriately cropped as in `samples/imgs`. You can also preprocess your images with `image_preprocess.py`.



#### License and Citation

```
@InProceedings{wang2021one,
author = Suzhen Wang, Lincheng Li, Yu Ding, Xin Yu
title = {One-shot Talking Face Generation from Single-speaker Audio-Visual Correlation Learning},
booktitle = {AAAI 2022},
year = {2022},
}
```



#### Acknowledgement

This codebase is based on [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) and [imaginaire](https://github.com/NVlabs/imaginaire), thanks for their contributions.





