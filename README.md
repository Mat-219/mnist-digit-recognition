1. Potrzebujesz Pythona 3.12 (na 3.13+ TensorFlow się nie zainstaluje) 
2. Zrób venv i zainstaluj paczki (Windows):
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
3. Odpal trening:
python -m src.train

jakieś 2-3 minuty, wyniki i wykresy są w folderze results/.
nie patrz na warningi w terminalu, to nie błędy