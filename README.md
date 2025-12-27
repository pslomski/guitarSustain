# modules
python3 -m pip install pytest

# unit tests
python3 -m pytest -q

# ffmepg
ffmpeg -i input.wav -ar 16000 -ac 1 -acodec pcm_s16le output.wav

# guitarSustain

guitar        t_half [ms]
telecaster    531
squire        677
gibson        720
hagstrom      820
squire e flat 844
witkowski     976
