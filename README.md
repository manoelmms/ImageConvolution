# Implementação Paralela de Convolução em Imagens

Este repositório contém a implementação de algoritmos de convolução de imagens utilizando técnicas de programação concorrente, em Python e C. O projeto foi desenvolvido como parte da disciplina de **Programação Concorrente (2024/1)**, oferecida pelo **Instituto de Computação da Universidade Federal do Rio de Janeiro (UFRJ)**.

## Descrição do Projeto

A convolução é uma operação matemática essencial no processamento de imagens, responsável pela aplicação de filtros para a detecção de padrões, como em redes neurais convolucionais (CNNs). No entanto, o cálculo sequencial da convolução é computacionalmente custoso, especialmente para imagens de grandes dimensões. Assim, implementamos versões paralelas da convolução, utilizando tanto Python quanto C, a fim de comparar a eficiência e o desempenho entre as duas linguagens.

## Estruturas e Algoritmos

O projeto inclui as seguintes abordagens para a implementação concorrente da convolução:

- **Python:**
  - Implementação concorrente usando processos com a biblioteca `multiprocessing`.
  - Implementação com a biblioteca **Numba** para paralelização otimizada com o compilador LLVM.
- **C:**
  - Implementação concorrente usando threads, com divisão em blocos para otimização de cache.

## Estratégias de Paralelização

1. **Python - Divisão Dinâmica**: Cada processo calcula a convolução de forma independente para cada pixel.
2. **Python - Divisão em Blocos**: A imagem é dividida em blocos, e cada processo é responsável por um bloco.
3. **Numba**: Implementação otimizada utilizando paralelização automática com `@jit` e `prange`.
4. **C - Divisão em Blocos**: Semelhante à versão em Python, mas implementada em C, aproveitando melhor a gestão de memória e threads.

## Testes e Avaliação

Foram criados testes automatizados para verificar a corretude e o desempenho das implementações. As imagens de teste utilizadas possuem dimensões:

- 333x333
- 500x500
- 1024x768

Além disso, realizamos testes de desempenho com diferentes quantidades de processos/threads (1, 3, 7, 20) para comparar o ganho de velocidade em relação à versão sequencial.

## Resultados

### Comparação de Desempenho

Os testes de desempenho mostraram que a implementação com **Numba** foi a mais eficiente, superando até mesmo a versão em C. A versão em C apresentou resultados significativamente mais rápidos do que o Python nativo, que foi em torno de 100 vezes mais lento.

### Conclusões

Apesar das limitações do Python em termos de gerenciamento de threads (devido ao Global Interpreter Lock - GIL), a utilização da biblioteca **Numba** permitiu atingir uma performance comparável à linguagem C. Esta biblioteca se mostrou extremamente eficiente para algoritmos numéricos que necessitam de grande poder computacional.

## Como Executar

### Requisitos

- Python 3.x
- GCC para compilar o código em C
- Bibliotecas: `Numba`, `multiprocessing`, `numpy`, `opencv-python`

### Instruções
#### Python

1. Execute o seguinte comando no terminal e escolha os modos:

   ```bash
   python main.py <input_image> <output_image>
   ```
> 💡 Caso queira executar a rotina de teste embutida no projeto, que irá testar os códigos tanto em C quanto em Python, execute o seguinte comando:

   ```bash
   python3 test.py
   ```
#### C
1. Caso deseje usar a versão em C, basta compilar a main usando:
   ```bash
   gcc -o main ./C/main.c ./C/convolution.c -lpthread -Wall
   ```
2. E execute o seguinte comando:
   ```bash
   ./main <matriz.bin> <kernel.bin> <n_threads> <matriz_saida>
   ```
> Atenção: Esta versão somente aceita arquivos binários de imagem e de kernel, o que é gerado pelo código na função image.py e kernel.py. Por conta disso, recomendamos seu uso apenas para comparação com sua versão em Python.



