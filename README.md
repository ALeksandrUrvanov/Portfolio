## Портфолио

### **Выполненные проекты**

<table>
  
<tr>
  <th>№</th>
  <th>Название</th>
  <th>Задача</th>
  <th>Инструменты</th>
  <th>Итоги</th>
</tr> 


<tr>
  <td>1</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/ceres_gpt"> AI-ассистент  </a> </td>
  <td>1. Создать AI-ассистента в виде консольного приложения и Telegram бота.</td>
  <td> Python, OpenAI API (GPT-4-mini), aiogram 3.x, LangChain, Hugging Face Transformers, FAISS, AsyncIO  </td>
  <td> AI-ассистент: https://t.me/ceres_assistant_bot?start </td>
</tr>

<tr>
  <td>2</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/asr_research"> Исследование в сфере ASR  </a> </td>
  <td>1. Написать программу распознования голосовых фраз. 2. Используя платформу Huggingface и готовую библиотеку Python написать программу для разделения аудиодорожки на отдельные персоны и распознавания текста, сказанного каждой персоной на аудио.
 </td>
  <td> vosk, whisper, pyannote, nltk, speech_recognition, gTTS, AudioSegment, numpy, SbertPuncCase, gc  </td>
  <td> Модель whisper/pyannote/speaker-diarization-3.1 для диалога на анлийско языке показала наилучший результат. </td>
</tr>

<tr>
  <td>3</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/define_train_number"> Распознание номеров грузовых вагонов по фотографии  </a> </td>
  <td>Предоставлена выборка фотографий номеров грузовых вагонов с аннотацией. Необходимо построить модель распознавания изображений.</td>
  <td> YoloV8, TrOCRProcessor, VisionEncoderDecoderModel, Roboflow, PIL, IPython, seaborn, numpy, pandas, matplotlib  </td>
  <td> Модель YoloV8m показывает хорошие результаты, производительность модели на разных уровнях сложности обнаружения на тестовой выборке mAP50-95 = 0.87, mAP50 = 0.995. </td>
</tr>

<tr>
  <td>4</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predict_cost_houses"> Предсказание стоимости строящегося жилья </a> </td>
  <td> Построение модели предсказания стоимости жилья на основе имеющихся данных </td>
  <td> pandas, numpy, phik, matplotlib, sklearn, seaborn, RandomForestRegressor, Ridge, CatBoostRegressor, LGBMRegressor </td>
  <td> Выбрана лучшая модель LightGBMRegressor, результат метрики на обучающей части R2 = 0.8754, на валидационной части R2 = 0.8781.
Прогнозы в среднем ошибаются на 9.55%. </td>
</tr>

<tr>
  <td>5</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/define_music_genre"> Определение жанра по изображению обложки музыкального диска </a> </td>
  <td>Очевидно, что оформление музыкального альбома как-то связано с его содержанием. Но насколько связано? Как подтвердить это, опираясь на данные? И чем это может быть полезно?</td>
  <td> pandas, sklearn, pytorch, fastai, resnet50  </td>
  <td> Samples avg (f1-score) = 0.82 средняя оценка, считая все категории равноправными. В целом модель обучилась хорошо, но из-за дисбаланса классов метрика хуже, чем могла бы быть. Рекомендацией является увеличение размера текущего набора данных, а точнее малочисленных класоов с f1-score < 0,6 : disco, jazz, pop. Это позволит улучшить метрику и качество работы модели. </td>
</tr>

<tr>
  <td>6</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/tutor_scam"> Построение ML-продукта для выявления и оптимизации платежей преподавателей сервиса Repetit.ru </a> </td>
  <td>Заказчику нужно как можно раньше понять, что репетитор недобросовестный или мошенник, чтобы отключить его от сервиса и отдавать заявки ответственным репетиторам.</td>
  <td> pandas, numpy, sklearn, phik, pipeline, RandomForestClassifier, CatBoostClassifier, LightGBM  </td>
  <td> По метрике F1 лучше показала себя модель RandomForestClassifier, но анализируя матрицу ошибок сделал вывод, что модель CatBoostClassifier более полезна для предсказания недобросовестных репетиторов. Для улучшения качества метрики рекомендуется увеличить тренировочную выборку. </td>
</tr>

<tr>
  <td>7</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/subtitles_english_level"> Классификация фильмов по уровню владения английского языка </a> </td>
  <td>Разработать модель соотносящая фильмы к определенному уровню владения английского языка. </td>
  <td> pandas, numpy, sklearn, ntlk, pipeline, GridSearchCV, MultinomialNB, SGDClassifier  </td>
  <td> Использовал модель SGDClassifier c Accuracy = 0.951220. Для заказчика однозначно рекомендую использовать эту модель для определения уровня английского языка по субтитрам.</td>
</tr>

<tr>
  <td>8</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/search_for_images_on_demand"> Разработка демонстрационной версии поиска изображений по текстовому запросу </a> </td>
  <td> Разработать нейронную сеть, которая получит векторное представление изображения, векторное представление текста, а на выходе выдаст число от 0 до 1 — что покажет, насколько текст и картинка подходят друг другу. </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, tensorflow, keras_nlp, torch, glob, nltk, AutoModel, AutoTokenizer, GridSearchCV, Ridge </td>
  <td> С помощью сети ResNet50 векторизировали фото, для векторизации текстов использовали DistilBert. Обучили полносвязную нейронную сеть, RMSE = 21.97%. </td>
</tr>

<tr>
  <td>9</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predictions_temperature_stars"> Прогнозирование температуры звезды </a> </td>
  <td> Разработать нейронную сеть, которая поможет предсказывать абсолютную температуру на поверхности звезды. Достичь заданной метрики RMSE < 4500. </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, torch </td>
  <td> Обучили нейросеть. Достигли заданной метрики RMSE < 4500, с результатом 4491. </td>
</tr>

<tr>
  <td>10</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predictions_orders_taxi"> Прогнозирование заказов такси </a> </td>
  <td> Разработать модель прогнозирования количество заказов такси на следующий час. Значение метрики RMSE на тестовой выборке должно быть не больше 48. </td>
  <td> pandas, numpy, matplotlib, sklearn, statsmodels, RandomForestRegressor, Ridge, CatBoostRegressor, LGBMRegressor </td>
  <td> Использовал модель RandomForestRegressor. RMSE на тестовой выборке =  46. </td>
</tr>

<tr>
  <td>11</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/used_cars_price"> Построение модели предсказания стоимости автомобиля на вторичном рынке </a> </td>
  <td> Разработать модель предсказания стоимости автомобиля на вторичном рынке. </td>
  <td> pandas, numpy, scipy, sklearn, seaborn, phik, CatBoostRegressor, XGBRegressor, Ridge </td>
  <td> Разработал модель CatBoostRegressor показатель MAPE = 0.1977. </td>
</tr>

<tr>
  <td>12</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/churn_bank_customers"> Построение модели прогнозирования оттока клиентов банка </a></td>
  <td> Построить модель, способную спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. </td>
  <td> pandas, sklearn, matplotlib </td>
  <td> Модель прогнозирования оттока клиентов банка достигла заданных показателей метрик  F1 = 0,60; AUC-ROC = 0,85.</td>
</tr>

<tr>
  <td>13</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/churn_telecom_customers"> Прогнозирование оттока клиентов компании "Теледом" </a></td>
  <td> Обучить модель для прогноза оттока клиентов. </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, lightgbm, xgboost, shap, phik, torch </td>
  <td> NeuralNetwork опередила остальные модели. На тестовой выборке NeuralNetwork показала ROC-AUC: 0.85. Что удовлетворяет поставленным требованиям компании оператора связи.</td>
</tr>

<tr>
  <td>14</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/churn_hotel_customers"> Прогнозирование отказа клиентов отеля от брони </a></td>
  <td> Обучить модель для прогнозирования отказа клиентов отеля от брони. </td>
  <td> pandas, numpy, scipy, sklearn, matplotlib, seaborn </td>
  <td>  Модель дерева решений показала хорошие результаты на тестовой выборке ROC-AUC = 0.9198 . Модель принесёт компании выручку: 8 841 480 рублей, что является целесообразным действием при затратах на разработку системы прогнозирования 400 000 рублей.  </td>
</tr>

<tr>
  <td>15</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/recommend_tariffs"> Рекомендация тарифов телеком компании </a></td>
  <td> Построить модель для задачи классификации, которая выберет подходящий тариф с значением Accuracy не меньше 0.75. </td>
  <td> pandas, sklearn </td>
  <td> На тестовой выборке модель случайного леса показала Accuracy = 0.7791.  </td>
</tr>

<tr>
  <td>16</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/define_age_customers"> Определение возраста покупателей по фотографии </a></td>
  <td> Построить и обучить свёрточную нейронную сеть на датасете с фотографиями людей. Добиться значения MAE на тестовой выборке не больше 8. </td>
  <td> pandas, numpy, matplotlib, seaborn, tensorflow, keras </td>
  <td> Использовали архитектуру ResNet50. Test MAE: 5.8556.  </td>
</tr>

<tr>
  <td>17</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/define_negative_comments"> Классификация комментариев на позитивные и негативные </a></td>
  <td> Построить модель классификации комментариев на позитивные и негативные со значением метрики качества F1 не меньше 0.75. </td>
  <td> pandas, numpy, spacy, sklearn, ntlk, LGBMClassifier, CatBoostClassifier </td>
  <td> Лучшей моделью по требуемому параметру является LogisticRegression.  F1 на тестовой выборке = 0,7761.  </td>
</tr>

<tr>
  <td>18</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predictions_cost_flats"> Предсказание стоимости жилья </a></td>
  <td> Обучить модель линейной регрессии предсказывающую медианную стоимость дома в жилом массиве и сделайте предсказания на тестовой выборке.  </td>
  <td> pandas, numpy, seaborn, matplotlib, pyspark </td>
  <td> По результату исследования модель линейной регрессии с категориальными переменными показала лучшие показатели RMSE = 73713. </td>
</tr>

<tr>
  <td>19</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/risk_cars_accident"> Oценить риск ДТП по выбранному маршруту движения </a></td>
  <td> Создать модель предсказания ДТП.  </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, lightgbm, shap, phik </td>
  <td> Использовал модель LightGBM и получил метрику recall = 72.9%, precision = 74.3%,  f1 = 72.7%. </td>
</tr>

<tr>
  <td>20</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/well_selection"> Выбор локации для разработки скважин </a></td>
  <td> С помощью машинного обучения выбрать районы, которые подходят для разработки новых скважин по экономическим показателям.  </td>
  <td> pandas, numpy, scipy, sklearn, matplotlib  </td>
  <td> По итогу расчета только один регион был принят в качестве экономически надежного для разработки местрождений. Необходимый объем сырья для безубыточной разработки 1 скважины в тыс. баррелей 111.11.  </td>
</tr>

<tr>
  <td>21</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predictions_cost_cars"> Определение стоимости автомобилей </a></td>
  <td> Построить модель для определения стоимости. Заказчику важны: качество предсказания, скорость предсказания, время обучения. </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, CatBoostRegressor, LGBMRegressor </td>
  <td> Лучшей моделью по трем параметрам качество, скорость обучения и скорость предсказания можно считать CatBoostRegressor. RMSE для модели CatBoostRegressor на тестовой выборке 1328.  </td>
</tr>


</table>
