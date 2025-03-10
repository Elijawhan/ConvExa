pysetup:
	python3 -m venv ".venv"
	source ".venv/bin/activate"
	python3 -m pip install -r requirements.txt

build: 
	rm -rf ./build
	cmake -S . -B build
	cd build && $(MAKE)

validate: pysetup build
	mv build/convexa*.so ./validation/convexa.so


clean:
	rm -rf ./build
