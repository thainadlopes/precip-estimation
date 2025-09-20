# Estimativa e Detecção de Precipitação com XGBoost

Este projeto utiliza o conjunto de dados **[SatRain](https://ipwgml.readthedocs.io/en/latest/intro.html)**, que combina observações de satélite e radar para o treinamento e a validação de modelos de Machine Learning.  
Para facilitar a implementação, também foi utilizada a biblioteca oficial [SatRain](https://github.com/ipwgml/satrain).

## Descrição
O código implementa um pipeline composto pelas seguintes etapas:
- **Pré-processamento** dos dados SatRain;
- **Treinamento** de um modelo baseado em **XGBoost** para:
  - Estimativa de precipitação;
  - Detecção de precipitação;
- **Avaliação** do desempenho do modelo por meio de métricas estatísticas como: Bias, MSE, MAE, SMAPE, Coeficiente de correlação linear e Resolução efetiva.
