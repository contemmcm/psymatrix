cohmetrix:
	docker build --no-cache --build-arg license=${COHMETRIX_LICENSE} -t cohmetrix -f Dockerfile.cohmetrix .
	
taaco:
	wget https://github.com/LCR-ADS-Lab/TAACO/archive/refs/heads/main.zip
	unzip main.zip
	rm main.zip
	mv TAACO-main taaco
	python -m spacy download en_core_web_sm

data/topics/lda:
	gdown https://drive.google.com/uc?id=1dh6OdAA4xULU1TN0FDuo4sxSUvX2DLTZ
	mkdir -p data/topics
	unzip lda.zip -d data/topics
	rm lda.zip

data/vae:
	gdown https://drive.google.com/uc?id=16r5mG7_Rv6qYpRAdTZG1yKaO_PluMh3i
	tar -xvf vae.tar.gz 
	rm vae.tar.gz

data/neural_net:
	gdown https://drive.google.com/uc?id=120wsfsepJQWVtR5If1-WntX3YgHdaqj2
	tar -xvf neural_net.tar.gz
	rm neural_net.tar.gz

download: taaco data/topics/lda data/vae data/neural_net
