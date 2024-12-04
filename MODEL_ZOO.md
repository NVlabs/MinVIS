# MinVIS Model Zoo

## Introduction

This file documents a collection of trained MinVIS models.
The numbers in this page are for the specific checkpoints and are different from the paper, which are averaged from multiple runs. The "Config" column contains a link to the config file. Running `train_net_video.py --num-gpus $num_gpus` with this config file will train a model with the same setting. ResNet-50 results are trained with 8 GPUs and Swin-L results are trained with 16 GPUs.

## Video Instance Segmentation
### YouTubeVIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Annotation</th>
<th valign="bottom">AP</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: R50 -->
 <tr><td align="center">R50</td>
<td align="center">100</td>
<td align="center">100%</td>
<td align="center">47.3</td>
<td align="center"><a href="configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis19_R50.pth">model</a></td>
</tr>
<!-- ROW: Swin-L -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">100%</td>
<td align="center">62.0</td>
<td align="center"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis19_swin_large.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 10% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">10%</td>
<td align="center">60.9</td>
<td align="center"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r10.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis19_swin_large_r10.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 5% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">5%</td>
<td align="center">59.7</td>
<td align="center"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r5.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis19_swin_large_r5.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 1% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">1%</td>
<td align="center">58.9</td>
<td align="center"><a href="configs/youtubevis_2019/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r1.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis19_swin_large_r1.pth">model</a></td>
</tr>
</tbody></table>


### YouTubeVIS 2021

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Annotation</th>
<th valign="bottom">AP</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: R50 -->
 <tr><td align="center">R50</td>
<td align="center">100</td>
<td align="center">100%</td>
<td align="center">43.9</td>
<td align="center"><a href="configs/youtubevis_2021/video_maskformer2_R50_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis21_R50.pth">model</a></td>
</tr>
<!-- ROW: Swin-L -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">100%</td>
<td align="center">55.4</td>
<td align="center"><a href="configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis21_swin_large.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 10% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">10%</td>
<td align="center">54.8</td>
<td align="center"><a href="configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r10.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis21_swin_large_r10.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 5% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">5%</td>
<td align="center">54.5</td>
<td align="center"><a href="configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r5.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis21_swin_large_r5.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 1% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">1%</td>
<td align="center">52.6</td>
<td align="center"><a href="configs/youtubevis_2021/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r1.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ytvis21_swin_large_r1.pth">model</a></td>
</tr>
</tbody></table>

### Occluded VIS

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Annotation</th>
<th valign="bottom">AP</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: R50 -->
 <tr><td align="center">R50</td>
<td align="center">100</td>
<td align="center">100%</td>
<td align="center">26.7</td>
<td align="center"><a href="configs/ovis/video_maskformer2_R50_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ovis_R50.pth">model</a></td>
</tr>
<!-- ROW: Swin-L -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">100%</td>
<td align="center">41.7</td>
<td align="center"><a href="configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ovis_swin_large.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 10% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">10%</td>
<td align="center">37.1</td>
<td align="center"><a href="configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r10.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ovis_swin_large_r10.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 5% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">5%</td>
<td align="center">35.7</td>
<td align="center"><a href="configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r5.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ovis_swin_large_r5.pth">model</a></td>
</tr>
<!-- ROW: Swin-L 1% -->
 <tr><td align="center">Swin-L (IN21k)</td>
<td align="center">200</td>
<td align="center">1%</td>
<td align="center">31.8</td>
<td align="center"><a href="configs/ovis/swin/video_maskformer2_swin_large_IN21k_384_bs32_8ep_frame_r1.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/deahuang/MinVIS-Model-Zoo/blob/main/minvis_ovis_swin_large_r1.pth">model</a></td>
</tr>
</tbody></table>
