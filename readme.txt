(cd whisperX/whisperx/;ln -s ../../transcribe_fastapi.py )


conda create -n whisperx_env python=3.11 -y
conda activate whisperx_env
pip install --upgrade pip setuptools wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchaudio==2.2.2 pyannote.audio==3.1.1 "numpy<2.0"
pip install -r whisperX/requirements.txt
pip install .
pip install uvicorn fastapi python-multipart




python -m whisperx.transcribe_fastapi --model tiny --output_format txt --output_dir out/  --diarize  ../../audio_files/sample.m4a ../../audio_files/sample.m4a



python -m whisperx.transcribe_fastapi --model tiny --output_format txt --output_dir out/  --diarize  --daemon 
curl -X POST "http://localhost:8000/transcribe?filename=../../audio_files/sample.m4a&mem_verbose=true"

IN_DIR=/home/ntg/eeas/whisper/2-get-a-docker/audio_files/
IN_File=s10feb25.mp4

P1=/home/ntg/eeas/whisper/2-get-a-docker/docker-whisperX/whisperX/
P2=/home/ntg/eeas/whisper/2-get-a-docker/
echo P1: ${P1}
echo P2: ${P2}
echo cache: ${P1}whisper_cache

## build dockerfile:
cd ${P1}docker-whisperX/
docker build --target no_model -t whisperx .
## export it to tar
docker save -o whisperx_no_model.tar whisperx
## or export just layer to zip
docker create --name whisperx_temp whisperx
docker export whisperx_temp | gzip > whisperx_no_model_export.tar.gz

#then import:
docker load -i whisperx_no_model.tar
#or:
gunzip -c whisperx_no_model_export.tar.gz | docker import - whisperx



docker run --gpus all -it 
  -v ${P1}:/app -v ${IN_DIR}:/in \
  -v ${P2}whisper_cache:/.cache -v ${P2}whisper_cache:/home/1001/.cache \
  -v ${P1}out:/out whisperx -- --model large-v2 \
  --output_format srt --output_dir out/ \
  --diarize /in/"${IN_File}"

