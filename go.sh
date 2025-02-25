MODEL_DIR=/home/ntg/eeas/whisper/2-get-a-docker/whisper_cache

echo MODEL_DIR: ${MODEL_DIR}

echo docker run --gpus all -it \
  -v ${MODEL_DIR}:/.cache -v ${MODEL_DIR}:/home/1001/.cache \
  whisperx -- --model large-v2 --daemon \
  --diarize 
echo
echo
echo
echo
docker run --gpus all -it \
  -v ${MODEL_DIR}:/.cache -v ${MODEL_DIR}:/home/1001/.cache \
  whisperx -- --model large-v2 --daemon \
  --diarize 

