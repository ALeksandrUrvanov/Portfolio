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
  <td><a href = "https://github.com/ALeksandrUrvanov/subtitles_english_level"> Классификация фильмов по уровню владения английского языка </a> </td>
  <td>Разработать модель соотносящая фильмы к определенному уровню владения английского языка. </td>
  <td> pandas, numpy, sklearn, ntlk, pipeline, GridSearchCV, MultinomialNB, SGDClassifier  </td>
  <td> Использовал модель SGDClassifier c Accuracy = 0.951220. Для заказчика однозначно рекомендую использовать эту модель для определения уровня английского языка по субтитрам.</td>
</tr>

<tr>
  <td>2</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/search_for_images_on_demand"> Разработка демонстрационной версии поиска изображений по текстовому запросу </a> </td>
  <td> Разработать нейронную сеть, которая получит векторное представление изображения, векторное представление текста, а на выходе выдаст число от 0 до 1 — что покажет, насколько текст и картинка подходят друг другу. </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, tensorflow, keras_nlp, torch, glob, nltk, AutoModel, AutoTokenizer, GridSearchCV, Ridge </td>
  <td> С помощью сети ResNet50 векторизировали фото, для векторизации текстов использовали DistilBert. Обучили полносвязную нейронную сеть, RMSE = 21.97%. </td>
</tr>

<tr>
  <td>3</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predictions_temperature_stars"> Прогнозирование температуры звезды </a> </td>
  <td> Разработать нейронную сеть, которая поможет предсказывать абсолютную температуру на поверхности звезды. Достичь заданной метрики RMSE < 4500. </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, torch </td>
  <td> Обучили нейросеть. Достигли заданной метрики RMSE < 4500, с результатом 4491. </td>
</tr>

<tr>
  <td>4</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predictions_orders_taxi"> Прогнозирование заказов такси </a> </td>
  <td> Разработать модель прогнозирования количество заказов такси на следующий час. Значение метрики RMSE на тестовой выборке должно быть не больше 48. </td>
  <td> pandas, numpy, matplotlib, sklearn, statsmodels, RandomForestRegressor, Ridge, CatBoostRegressor, LGBMRegressor </td>
  <td> Использовал модель RandomForestRegressor. RMSE на тестовой выборке =  46. </td>
</tr>

<tr>
  <td>5</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/used_cars_price"> Построение модели предсказания стоимости автомобиля на вторичном рынке </a> </td>
  <td> Разработать модель предсказания стоимости автомобиля на вторичном рынке. </td>
  <td> pandas, numpy, scipy, sklearn, seaborn, phik, CatBoostRegressor, XGBRegressor, Ridge </td>
  <td> Разработал модель CatBoostRegressor показатель MAPE = 0.1977. </td>
</tr>

<tr>
  <td>6</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/churn_bank_customers"> Построение модели прогнозирования оттока клиентов банка </a></td>
  <td> Построить модель, способную спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. </td>
  <td> pandas, sklearn, matplotlib </td>
  <td> Модель прогнозирования оттока клиентов банка достигла заданных показателей метрик  F1 = 0,60; AUC-ROC = 0,85.</td>
</tr>

<tr>
  <td>7</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/churn_telecom_customers"> Прогнозирование оттока клиентов компании "Теледом" </a></td>
  <td> Обучить модель для прогноза оттока клиентов. </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, lightgbm, xgboost, shap, phik, torch </td>
  <td> NeuralNetwork опередила остальные модели. На тестовой выборке NeuralNetwork показала ROC-AUC: 0.85. Что удовлетворяет поставленным требованиям компании оператора связи.</td>
</tr>

<tr>
  <td>8</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/churn_hotel_customers"> Прогнозирование отказа клиентов отеля от брони </a></td>
  <td> Обучить модель для прогнозирования отказа клиентов отеля от брони. </td>
  <td> pandas, numpy, scipy, sklearn, matplotlib, seaborn </td>
  <td>  Модель дерева решений показала хорошие результаты на тестовой выборке ROC-AUC = 0.9198 . Модель принесёт компании выручку: 8 841 480 рублей, что является целесообразным действием при затратах на разработку системы прогнозирования 400 000 рублей.  </td>
</tr>

<tr>
  <td>9</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/recommend_tariffs"> Рекомендация тарифов телеком компании </a></td>
  <td> Построить модель для задачи классификации, которая выберет подходящий тариф с значением Accuracy не меньше 0.75. </td>
  <td> pandas, sklearn </td>
  <td> На тестовой выборке модель случайного леса показала Accuracy = 0.7791.  </td>
</tr>

<tr>
  <td>10</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/define_age_customers"> Определение возраста покупателей по фотографии </a></td>
  <td> Построить и обучить свёрточную нейронную сеть на датасете с фотографиями людей. Добиться значения MAE на тестовой выборке не больше 8. </td>
  <td> pandas, numpy, matplotlib, seaborn, tensorflow, keras </td>
  <td> Использовали архитектуру ResNet50. Test MAE: 5.8556.  </td>
</tr>

<tr>
  <td>11</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/define_negative_comments"> Классификация комментариев на позитивные и негативные </a></td>
  <td> Построить модель классификации комментариев на позитивные и негативные со значением метрики качества F1 не меньше 0.75. </td>
  <td> pandas, numpy, spacy, sklearn, ntlk, LGBMClassifier, CatBoostClassifier </td>
  <td> Лучшей моделью по требуемому параметру является LogisticRegression.  F1 на тестовой выборке = 0,7761.  </td>
</tr>

<tr>
  <td>12</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predictions_cost_flats"> Предсказание стоимости жилья </a></td>
  <td> Обучить модель линейной регрессии предсказывающую медианную стоимость дома в жилом массиве и сделайте предсказания на тестовой выборке.  </td>
  <td> pandas, numpy, seaborn, matplotlib, pyspark </td>
  <td> По результату исследования модель линейной регрессии с категориальными переменными показала лучшие показатели RMSE = 73713. </td>
</tr>

<tr>
  <td>13</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/risk_cars_accident"> Oценить риск ДТП по выбранному маршруту движения </a></td>
  <td> Создать модель предсказания ДТП.  </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, lightgbm, shap, phik </td>
  <td> Использовал модель LightGBM и получил метрику recall = 72.9%, precision = 74.3%,  f1 = 72.7%. </td>
</tr>

<tr>
  <td>14</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/well_selection"> Выбор локации для скважины </a></td>
  <td> С помощью машинного обучения выбрать районы, которые подходят для разработки новых скважин по экономическим показателям.  </td>
  <td> pandas, numpy, scipy, sklearn, matplotlib  </td>
  <td> По итогу расчета только один регион был принят в качестве экономически надежного для разработки местрождений. Необходимый объем сырья для безубыточной разработки 1 скважины в тыс. баррелей 111.11.  </td>
</tr>

<tr>
  <td>15</td>
  <td><a href = "https://github.com/ALeksandrUrvanov/predictions_cost_cars"> Определение стоимости автомобилей </a></td>
  <td> Построить модель для определения стоимости. Заказчику важны: качество предсказания, скорость предсказания, время обучения. </td>
  <td> pandas, numpy, sklearn, matplotlib, seaborn, CatBoostRegressor, LGBMRegressor </td>
  <td> Лучшей моделью по трем параметрам качество, скорость обучения и скорость предсказания можно считать CatBoostRegressor. RMSE для модели CatBoostRegressor на тестовой выборке 1328.  </td>
</tr>


</table>
