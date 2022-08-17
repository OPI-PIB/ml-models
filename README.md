# Modele uczenia maszynowego udostępnione przez OPI PIB

## Neuronowe modele języka

### RoBERTa
Zestaw polskich neuronowych modeli języka opartych na architekturze Transformer, uczonych metodą maskowanego modelowania języka (MLM) przy wykorzystaniu technik opisanych w publikacji [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). Udostępniamy modele w dwóch rozmiarach - base (mniejsze) oraz large (większe). Mniejsze modele są sieciami neuronowymi liczącymi około 100 milionów parametrów, większe - 350 milionów. Model large oferuje wyższą jakość predykcji w praktycznych zastosowaniach, ale wymaga także większych zasobów obliczeniowych. Do wytrenowania modeli użyto dużych korpusów tekstów w języku polskim - od 20 do około 200 GB. Każdy model udostępniony został w dwóch wariantach pozwalających na odczytanie ich w popularnych bibliotekach do uczenia maszynowego: [Fairseq](https://github.com/pytorch/fairseq) oraz [Hugginface Transformers](https://github.com/huggingface/transformers).

Modele w wersji Fairseq: [base (wersja 1)](https://share.opi.org.pl/s/YammFDDFyymxHjA), [base (wersja 2)](https://share.opi.org.pl/s/X78QyWBXmbTmWTr), [large (wersja 1)](https://share.opi.org.pl/s/TBM8q5Bzrqaa5XF), [large (wersja 2)](https://share.opi.org.pl/s/zwK4mofafDtgBx2)

Modele w wersji Huggingface Transformers: [base (wersja 1)](https://share.opi.org.pl/s/j9A9Fmij6smDTe8), [base (wersja 2)](https://share.opi.org.pl/s/JonE4qDDjzsQAtT), [large (wersja 1)](https://share.opi.org.pl/s/RAmxCTKDNY4naWe), [large (wersja 2)](https://share.opi.org.pl/s/FTpq7ceAgdeyR5k)

### BART
Neuronowy model języka typu Transformer, wykorzystujący architekturę enkoder-dekoder. Model był uczony na zbiorze tekstów w języku polskim liczącym ponad 200 GB, przy wykorzystaniu metody opisanej w publikacji [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461). Model może być dostosowany do rozwiązywania zadań predykcyjnych, jednak jego główym zastosowaniem są zadania typu sequence to sequence, czyli takie, w których zarówno wejściem jak i wyjściem modelu jest tekst (np. tłumaczenie maszynowe, chatboty). Model udostępniony został w dwóch wariantach pozwalających na odczytanie ich w popularnych bibliotekach do uczenia maszynowego: [Fairseq](https://github.com/pytorch/fairseq) oraz [Hugginface Transformers](https://github.com/huggingface/transformers).

Pliki do pobrania: [model w wersji Fairseq](https://share.opi.org.pl/s/aw6o2g7joKS8m6D), [model w wersji Huggingface Transformers](https://share.opi.org.pl/s/nHPT3Ln7SBRyb5M)

### GPT-2
Neuronowy model języka oparty na architekturze Transformer, uczony metodą autogregresyjnego modelowania języka. Architektura sieci neuronowych jest zgodna z angielskojęzycznymi modelami GPT-2, opisanymi w publikacji [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Udostępniamy modele w dwóch rozmiarach - medium  (średni) liczący około 350 milionów parametrów oraz large (duży) liczący około 700 milionów parametrów. Pliki zostały zapisane w formacie pozwalającym na wczytanie ich w bibliotece [Fairseq](https://github.com/pytorch/fairseq).

Pliki do pobrania: [model medium](https://share.opi.org.pl/s/9p32SjLsASgepqz), [model large](https://share.opi.org.pl/s/TGXs2CytKnTbjNx)

### ELMo
ELMo jest modelem języka opartym na rekurencyjnych sieciach neuronowych typu LSTM (Long Short-Term Memory), zaproponowanym w publikacji [Deep contextualized word representations](https://arxiv.org/abs/1802.05365). Udostępniony model dla języka polskiego może być odczytany przy pomocy biblioteki [AllenNLP](https://github.com/allenai/allennlp).

Pliki do pobrania: [model](https://share.opi.org.pl/s/KrKRTytyQp7yka9)

## Statyczne reprezentacje słów

### Word2Vec
Klasyczne wektorowe reprezentacje słów dla języka polskiego, wytrenowane przy użyciu metody zaproponowanej w publikacji [Distributed Representations of Words and Phrases
and their Compositionality](https://arxiv.org/abs/1310.4546). Do uczenia wektorów użyto dużego korpusu tekstów w języku polskim. Zbiór zawiera około 2 milionów słów, w tym słowa występujące przynajmniej 3 razy w korpusie oraz inne zdefiniowane kategorie symboli (znaki interpunkcyjne, numery od 0 do 10 000, polskie imiona i nazwiska). Wektory zostały zapisane w formacie możliwym do oczytania przy pomocy biblioteki [Gensim](https://radimrehurek.com/gensim/). Udostępniamy wektory o zróżnicowanej liczbie wymiarów: od reprezentacji 100 wymiarowych do 800 wymiarowych.

Pliki do pobrania: [100d](https://share.opi.org.pl/s/w7eTXQWeAJXX8tP), [300d](https://share.opi.org.pl/s/PnZD2Yck3jQT4ye), [500d](https://share.opi.org.pl/s/NMQXAjbi3yx7gZL), [800d](https://share.opi.org.pl/s/QTz8Jt2gbMmtnkx)


### GloVe
Wektorowe reprezentacje słów dla języka polskiego, wytrenowane przy użyciu metody [GloVe](https://aclanthology.org/D14-1162/) opracowanej na Uniwersytecie Stanforda. Do uczenia wektorów użyto dużego korpusu tekstów w języku polskim. Zbiór zawiera około 2 milionów słów, w tym słowa występujące przynajmniej 3 razy w korpusie oraz inne zdefiniowane kategorie symboli (znaki interpunkcyjne, numery od 0 do 10 000, polskie imiona i nazwiska). Wektory zostały zapisane w formacie tekstowym, możliwym do odczytania przy pomocy różnych bibliotek obsługujących tego typu modele. Udostępniamy wektory o zróżnicowanej liczbie wymiarów: od reprezentacji 100 wymiarowych do 800 wymiarowych.

Pliki do pobrania: [100d](https://share.opi.org.pl/s/qeWtsizPZxJZXCY), [300d](https://share.opi.org.pl/s/kzWtFTTWAnNnmS4), [500d](https://share.opi.org.pl/s/TEernXTfFco2EXt), [800d](https://share.opi.org.pl/s/MQ4LisDdagX5DWL)

### FastText
Model zawierający wektorowe reprezentacje słów oraz cząstek słów w języku polskim. Jego główną zaletą w stosunku do tradycyjnych, statycznych reprezentacji języka jest możliwość generowania nowych wektorów dla słów, które nie znajdują się w słowniku, na podstawie sumy reprezentacji ich cząstek. Model był trenowany na dużym korpusie tekstów w języku polskim, przy wykorzystaniu metody zaproponowanej w publikacji [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606). Zbiór zawiera około 2 milionów słów, w tym słowa występujące przynajmniej 3 razy w korpusie oraz inne zdefiniowane kategorie symboli (znaki interpunkcyjne, numery od 0 do 10 000, polskie imiona i nazwiska). Wektory zostały zapisane w formacie możliwym do oczytania przy pomocy biblioteki [Gensim](https://radimrehurek.com/gensim/). Udostępniamy wektory o zróżnicowanej liczbie wymiarów: od reprezentacji 100 wymiarowych do 800 wymiarowych.

Pliki do pobrania: [100d](https://share.opi.org.pl/s/JGwNPApL4NH2Lza), [300d](https://share.opi.org.pl/s/5cGH7xMiJg3FzEW), [500d](https://share.opi.org.pl/s/kgMqjCL7WM3zQ62), [800d (część 1)](https://share.opi.org.pl/s/o2e37A6KsZ4odtd), [800d (część 2)](https://share.opi.org.pl/s/a6926zpKPLy9Bq7)

## Modele tłumaczenia maszynowego

Polsko-angielskie i angielsko-polskie modele oparte na sieciach splotowych.
Modele służące do automatycznego tłumaczenia tekstów dla biblioteki [Fairseq](https://github.com/pytorch/fairseq), oparte na neuronowych sieciach splotowych (konwolucyjnych). Udostępniamy dwa modele: polsko-angielski i angielsko-polski. Do ich uczenia wykorzystane zostały dane dostępne w serwisie [OPUS](http://opus.nlpl.eu/). Każdy z modeli był trenowany na zbiorze liczącym ponad 40 milionów par składających się ze zdania i jego tłumaczenia.

Pliki do pobrania: [model polsko-angielski](https://share.opi.org.pl/s/ztGPz7q7aHk4CfH), [model angielsko-polski](https://share.opi.org.pl/s/GTW5n4KdiyFcaAq)
