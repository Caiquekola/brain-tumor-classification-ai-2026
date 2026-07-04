# Classificação de Tumores Cerebrais em Imagens de Ressonância Magnética com Inteligência Artificial

Projeto prático desenvolvido para a disciplina de Inteligência Artificial em 2026.

O projeto utiliza técnicas de Aprendizado de Máquina e Aprendizado Profundo para classificar imagens de ressonância magnética cerebral em quatro categorias:

* Glioma
* Meningioma
* Tumor hipofisário
* Sem tumor

## Objetivo

O objetivo deste trabalho é desenvolver e comparar modelos de Inteligência Artificial capazes de identificar corretamente a classe de uma imagem de ressonância magnética cerebral.

Serão realizados pelo menos oito experimentos, utilizando no mínimo duas técnicas diferentes de Aprendizado de Máquina.

Os experimentos incluem variações de:

* arquitetura dos modelos;
* número de neurônios e filtros;
* taxa de dropout;
* regularização L1 ou L2;
* data augmentation;
* taxa de aprendizado;
* tamanho do batch;
* parâmetros específicos de cada algoritmo.

## Dataset

O conjunto de dados utilizado é o **Brain Tumor MRI Dataset**, disponibilizado no Kaggle.

Dataset:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data

O dataset possui aproximadamente 7.200 imagens de ressonância magnética cerebral, organizadas nas seguintes classes:

| Classe     | Descrição                                |
| ---------- | ---------------------------------------- |
| Glioma     | Imagens que apresentam glioma            |
| Meningioma | Imagens que apresentam meningioma        |
| Pituitary  | Imagens que apresentam tumor hipofisário |
| No Tumor   | Imagens sem presença de tumor            |

## Tecnologias utilizadas

* Python
* Google Colab
* TensorFlow
* Keras
* Scikit-Learn
* NumPy
* Pandas
* Matplotlib
* OpenCV
* Seaborn
* Kaggle API

## Etapas do projeto

O desenvolvimento está dividido nas seguintes etapas:

1. Download e organização do dataset;
2. Análise exploratória das imagens;
3. Verificação da distribuição das classes;
4. Redimensionamento e normalização das imagens;
5. Separação dos dados de treinamento, validação e teste;
6. Implementação dos modelos;
7. Realização de pelo menos oito experimentos;
8. Avaliação dos modelos;
9. Comparação dos resultados;
10. Seleção da melhor abordagem.

## Pré-processamento

As imagens são preparadas antes do treinamento por meio das seguintes operações:

* redimensionamento;
* normalização dos valores dos pixels;
* codificação dos rótulos;
* separação em treinamento, validação e teste;
* aplicação opcional de data augmentation;
* redução de dimensionalidade para os modelos que necessitarem.

O data augmentation pode incluir:

* pequenas rotações;
* zoom;
* deslocamentos horizontais e verticais;
* alterações moderadas de contraste.

As transformações são aplicadas somente ao conjunto de treinamento.

## Modelos avaliados

O projeto compara pelo menos duas técnicas diferentes de classificação.

### Rede Neural Convolucional — CNN

A CNN aprende características diretamente das imagens por meio de camadas convolucionais.

Entre os parâmetros avaliados estão:

* quantidade de camadas convolucionais;
* quantidade de filtros;
* número de neurônios;
* dropout;
* regularização L2;
* Batch Normalization;
* data augmentation;
* taxa de aprendizado.

### Support Vector Machine — SVM

O SVM é utilizado como uma segunda técnica de classificação.

Antes do treinamento, as imagens podem passar pelas seguintes etapas:

1. redimensionamento;
2. conversão para escala de cinza;
3. transformação em vetores;
4. normalização com `StandardScaler`;
5. redução de dimensionalidade com PCA;
6. classificação utilizando SVM.

Entre os parâmetros avaliados estão:

* kernel linear;
* kernel RBF;
* valor de `C`;
* valor de `gamma`;
* quantidade de componentes do PCA;
* balanceamento das classes.

## Experimentos

A configuração inicial dos oito experimentos é apresentada abaixo.

| Experimento | Modelo | Principal variação                                   |
| ----------- | ------ | ---------------------------------------------------- |
| 1           | CNN    | Modelo base                                          |
| 2           | CNN    | Dropout de 20%                                       |
| 3           | CNN    | Batch Normalization e dropout de 50%                 |
| 4           | CNN    | Data augmentation, regularização L2 e Early Stopping |
| 5           | SVM    | Kernel linear e C igual a 1                          |
| 6           | SVM    | Kernel RBF, C igual a 1 e gamma scale                |
| 7           | SVM    | Kernel RBF e C igual a 10                            |
| 8           | SVM    | PCA, kernel RBF e balanceamento das classes          |

As configurações podem ser ajustadas durante o desenvolvimento, desde que sejam mantidos pelo menos oito experimentos e duas técnicas diferentes.

## Métricas de avaliação

Os modelos são avaliados utilizando as seguintes métricas:

### Acurácia

Proporção de previsões corretas em relação ao total de imagens avaliadas.

### Precisão

Proporção das previsões positivas que estavam corretas.

### Recall

Proporção das amostras positivas que foram corretamente identificadas.

### F1-Score

Média harmônica entre precisão e recall.

Como o problema possui quatro classes, serão utilizadas principalmente as médias macro e weighted.

Também serão analisados:

* matriz de confusão;
* métricas individuais de cada classe;
* curvas de treinamento e validação;
* loss de treinamento e validação;
* tempo de treinamento;
* ocorrência de overfitting.

## Resultados

A tabela será preenchida após a execução dos experimentos.

| Exp. | Modelo                    | Acurácia | Precisão macro | Recall macro | F1 macro |
| ---- | ------------------------- | -------: | -------------: | -----------: | -------: |
| 1    | CNN base                  |        - |              - |            - |        - |
| 2    | CNN com dropout           |        - |              - |            - |        - |
| 3    | CNN com BatchNorm         |        - |              - |            - |        - |
| 4    | CNN com augmentation e L2 |        - |              - |            - |        - |
| 5    | SVM linear                |        - |              - |            - |        - |
| 6    | SVM RBF                   |        - |              - |            - |        - |
| 7    | SVM RBF com C igual a 10  |        - |              - |            - |        - |
| 8    | SVM com PCA               |        - |              - |            - |        - |

## Critério para escolha do melhor modelo

A escolha da melhor abordagem não será baseada apenas na acurácia.

Também serão considerados:

* F1-Score macro;
* recall das classes que representam tumores;
* quantidade de falsos negativos;
* capacidade de generalização;
* diferença entre treinamento e validação;
* tempo de treinamento;
* comportamento apresentado na matriz de confusão.

Em um problema relacionado à medicina, falsos negativos são especialmente importantes, pois representam casos em que uma imagem com tumor pode ser classificada incorretamente como sem tumor.

Este projeto possui finalidade exclusivamente acadêmica e não deve ser utilizado como ferramenta de diagnóstico médico.

## Estrutura do repositório

```text
brain-tumor-mri-classification/
│
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb
│   ├── 02_pre_processamento.ipynb
│   ├── 03_experimentos_cnn.ipynb
│   └── 04_experimentos_svm.ipynb
│
├── resultados/
│   ├── metricas.csv
│   ├── graficos/
│   └── matrizes_confusao/
│
├── modelos/
│
├── relatorio/
│   └── relatorio_final.pdf
│
├── README.md
└── requirements.txt
```

O dataset não será armazenado diretamente no GitHub devido à quantidade e ao tamanho das imagens.

## Execução no Google Colab

Os experimentos foram desenvolvidos para execução no Google Colab.

Para utilizar aceleração por GPU:

1. Abra o notebook no Google Colab;
2. Acesse `Ambiente de execução`;
3. Selecione `Alterar o tipo de ambiente de execução`;
4. Escolha `GPU` como acelerador de hardware;
5. Salve a configuração.

Para verificar se uma GPU está disponível:

```python
import tensorflow as tf

print(tf.config.list_physical_devices("GPU"))
```

### Montagem do Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

### Download do dataset com a API do Kaggle

Primeiramente, envie o arquivo `kaggle.json` para o ambiente do Colab.

```python
from google.colab import files

files.upload()
```

Depois, configure as credenciais:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

Faça o download e descompacte o dataset:

```python
!kaggle datasets download \
    -d masoudnickparvar/brain-tumor-mri-dataset

!unzip -q brain-tumor-mri-dataset.zip \
    -d /content/brain-tumor-mri-dataset
```

O arquivo `kaggle.json` contém credenciais pessoais e não deve ser enviado para o GitHub.

Adicione ao arquivo `.gitignore`:

```text
kaggle.json
*.h5
*.keras
dataset/
dados/
__pycache__/
.ipynb_checkpoints/
```

## Como executar

1. Faça um fork ou clone deste repositório;
2. Abra os notebooks no Google Colab;
3. Ative o ambiente com GPU;
4. Faça o download do dataset;
5. Execute inicialmente a análise exploratória;
6. Execute o pré-processamento;
7. Execute os experimentos de CNN;
8. Execute os experimentos de SVM;
9. Analise e compare as métricas;
10. Preencha a tabela final de resultados.

## Relatório e apresentação

O relatório final apresentará:

* descrição do problema;
* análise do dataset;
* métodos de pré-processamento;
* modelos utilizados;
* configurações dos oito experimentos;
* códigos utilizados;
* métricas obtidas;
* matrizes de confusão;
* comparação dos modelos;
* análise da melhor abordagem;
* conclusões e possíveis melhorias.

O link para o vídeo de apresentação será adicionado após a gravação.

## Vídeo de apresentação

Link: `ADICIONAR LINK DO VÍDEO`

## Autor

**Caíque Augusto**

Projeto Prático de Inteligência Artificial — 2026.

## Aviso

Este projeto foi desenvolvido exclusivamente para fins acadêmicos.

Os modelos e resultados apresentados não devem ser utilizados para realizar diagnósticos, orientar tratamentos ou substituir a avaliação de profissionais da área médica.
