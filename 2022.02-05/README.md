# Задача обратной нормализации текста (Inverse Text Normalization)

## 14.02.2022-20.02.2022

Общение с научным руководителем по поводу дальнейшего направления работы.

## 21.02.2022-06.03.2022

Начал проведить ablation study для моделей на основе трансформера. Перебираю для него гиперпараметры.

## 27.03.2022-03.04.2022

Закончил проведить ablation study для моделей на основе трансформера.

Лучшее качество (WER, Word Error Rate) было достигнуто при размерности эмбеддингов 512, количестве слоев энкодера и декодера равному 3, размерности линейных слоев 512 и темпе обучения 1e-4 или 3e-4 в зависимости от модели.

## 04.04.2022-17.04.2022

Реализация моделей №№4-6 в режиме обучения с использованием меток из классификатора. Проведение экспериментов. По итогу получилось, что качество таких моделей сильно хуже, чем при обучении с правильными классами. Результаты в таблице ниже.

| Модель      | T-P   | T-T  | P-P  | P-T   |
| ----------- | ----- | ---- | ---- | ----- |
| RNN         | 19.92 | -    | -    | -     |
| Трансформер | 9.86  | -    | -    | -     |
| Модель №1   | 6.26  | -    | -    | -     |
| Модель №2   | 7.25  | 6.09 | 6.39 | 6.44  |
| Модель №3   | 7.47  | 6.01 | 6.27 | 6.24  |
| Модель №4   | 5.73  | 4.05 | 6.39 | 17.00 |
| Модель №5   | 5.34  | 2.74 | 6.38 | 17.13 |
| Модель №6   | 7.01  | 3.25 | 7.79 | 18.68 |

## 18.04.2022-01.05.2022

Реализация лучевого поиска (Beam Search). Проведение экспериментов над моделями. Есть улучшение качества при ширине поиска равном 3. Результаты в таблице ниже.

|   Модель    | Тип меток при обучении | Жадный поиск | Лучевой поиск (3) | Лучевой поиск (10) |
| :---------: | :--------------------: | :----------: | ----------------- | ------------------ |
|     RNN     |           N            |    19.92     | 19.32             | 19.29              |
| Трансформер |           N            |     9.86     | 9.59              | 9.57               |
|  Модель №1  |           N            |     6.26     | 6.06              | 6.06               |
|  Модель №2  |           T            |     7.25     | 7.08              | 7.08               |
|             |           P            |     6.39     | 6.15              | 6.13               |
|  Модель №3  |           T            |     7.47     | 7.33              | 7.35               |
|             |           P            |     6.30     | 6.10              | 6.12               |
|  Модель №4  |           T            |     5.73     | 5.64              | 5.64               |
|             |           P            |     6.39     | 6.38              | 6.38               |
|  Модель №5  |           T            |     5.34     | 5.33              | 5.34               |
|             |           P            |     6.38     | 6.37              | 6.37               |
|  Модель №6  |           T            |     7.01     | 6.85              | 6.85               |
|             |           P            |     7.79     | 7.51              | 7.52               |

## 01.05.2022-07.05.2022

Написание магистерской диссертации. Предзащита. Выложен код экспериментов.
