apt-get update
apt-get install -y libgl1 libglib2.0-0

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# cp LightGlue/ckpts/superpoint_lightglue.pth /root/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth
# cp LightGlue/ckpts/superpoint_v1.pth /root/.cache/torch/hub/checkpoints/