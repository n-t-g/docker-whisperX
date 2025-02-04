(cd whisperX/whisperx/;ln -s ../../transcribe_fastapi.py )


conda create -n whisperx_env python=3.11 -y
conda activate whisperx_env
pip install --upgrade pip setuptools wheel
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.2 torchaudio==2.2.2 pyannote.audio==3.1.1 "numpy<2.0"
pip install -r whisperX/requirements.txt
pip install .
pip install uvicorn fastapi python-multipart




python -m whisperx.transcribe_fastapi --model tiny --output_format txt --output_dir out/  --diarize  ../../audio_files/sample.m4a ../../audio_files/sample.m4a

