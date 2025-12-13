---
sidebar_position: 3
authors:
  - name: Daniel Bazo Correa
description: Fundamentos del Deep Learning.
title: Redes Neuronales de Grafos
toc_max_heading_level: 3
---

# Machine Learning with Graphs

## Lecture 1.1

- Social networks, economics, communications → graphs.  
  Better relations and representations between data.

- Types of networks and graphs:
  - **Natural graphs**: any structure whose nature is represented as a graph.
  - **Representations**: structures that we can interpret as a graph.

- More complex to process:
  - arbitrary sizes
  - complex topological structure
  - no order in the nodes
  - dynamics
  - multimodal characteristics

## Graph Learning

**Graph → Model (GNN) → Prediction**
- Node labels
- New links
- New graphs / subgraphs

## Node Representation

- Node \( u \)
- Function:
  
  \[
  f: u \rightarrow \mathbb{R}^d
  \]

  Function that maps node \( u \) to a vector (representation) of size \( d \).

## Lecture 1.2

### Diferentes tipos de tareas

1. A nivel de nodos  
2. Subgrafo (un grupo de nodos más pequeño que el grafo principal)  
3. A nivel de enlaces  
4. A nivel del grafo al completo  

### Por ejemplo

- **Clasificación de nodos**: predecir propiedades de un nodo.

- **Predicción de enlaces**: predecir enlaces faltantes → *Knowledge Graph Completion*.

- **Clasificación de grafos**: agregar categorías.

- **Clustering de nodos**.

- **Evoluciones o generación de grafos**  
  - p. ej. simulaciones

Aquí tienes el **texto extraído y estructurado en Markdown** a partir de la imagen:

```markdown
## Lecture 1.3

### Componentes de un grafo

- **Objetos**: nodos, vértices → \( N \)
- **Interacciones**: enlaces, *edges* → \( E \)
- **Sistema**: grafo, red → \( G(N, E) \)

> Representación matemática

### Definición de un grafo

Para crear un grafo tenemos que definir:

- **Nodos**
- **Enlaces**

En algunos casos solo existe una única posible representación, en otros puede haber múltiples representaciones.

> La importancia reside más en **cómo vamos a asignar los enlaces**.

### Tipos de grafos

Los grafos pueden ser:

- **Enlaces *undirected*** (sin dirección), por lo que existe simetría (recíprocos).

```

A — B

```

- **Enlaces *directed*** (con dirección).

```

A → B

```
```

Para grafos undirected: 


 

→ Grado del modo \((K_i)\): uno de
eulaceas adyacentes al modo i.
P: ej \(K_A = 4\) 

→ El promedio sería: 

\[
G_1(N, E) \quad \leftarrow \quad \frac{1}{K} = \langle K \rangle = \frac{1}{N} \sum_{i=1}^{N} K_i = \frac{2E}{N}
\]

* E: edges
* N: modos (Cada edge vale 2 

\[
P_i: \frac{1}{K_i} = \frac{2 \cdot 7}{6} = 2,3
\]

Para grafos directed: 


 

→ Grado modo directo 

→ Grado interno (in-degree):
\((K_i)\): uno eulaceas apuntando al modo 

→ Grado externo (out-degree) \((K_i^{out})\): uno enlace apuntando fuera del modo. 

\[
K_i^{in} \quad K_A = 3 \quad K_i^{out} = 4 \quad K = K_i^{in} + K_A = 4
\]

El promedio:
\[
\overline{K} = \frac{E}{2}
\]

