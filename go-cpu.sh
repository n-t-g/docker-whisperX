MODEL_DIR=/home/llmdev/code/asr-test/whisperx/whisper_cache

echo MODEL_DIR: ${MODEL_DIR}

echo docker run -d --name whisperx_cpu \
  -v ${MODEL_DIR}:/.cache -v ${MODEL_DIR}:/home/1001/.cache \
  whisperx -- --model large-v2 --daemon --device cpu --compute_type int8\
  --diarize $@
echo
echo
echo
echo
docker run -d --name whisperx_cpu \
  -v ${MODEL_DIR}:/.cache -v ${MODEL_DIR}:/home/1001/.cache \
  whisperx -- --model large-v2 --daemon --device cpu --compute_type int8\
  --diarize $@
