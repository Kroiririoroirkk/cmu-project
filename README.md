# CMU Summer 2024 uPNC Project

Eric Tao, 2024-06-10

This is the repository I (Eric Tao) have made for the code I have written as part of the Summer 2024 uPNC (Undergraduate Program in Neural Computation) at CMU (Carnegie Mellon University). The aim of this project is to investigate the functional basis of why neurons are connected in the patterns that they are. For example, [previous work](https://bmcbiol.biomedcentral.com/articles/10.1186/1741-7007-2-25) by Reigl et al., 2004 has found that certain types of network motifs such as bi-directionally connected pairs of neurons and transitive triangles of neurons are overrepresented inside a _C. elegans_ brain. How does the presence of these motifs in a neural network change its ability to perform various tasks? What tasks do networks with these motifs excel at? What tasks do networks with these motifs fail at? The answers to these questions would yield insight into the evolutionary goals of the brain and bridge the gap between network connectivity and functionality, analogous to Marr's implementational and computational levels.

The main notebook file for my exploration is `LDS_exploration.ipynb`. For posterity, I have also included some old code in the `old` folder which uses PyTorch. I have since decided to code the system from scratch for a better understanding of the internals and more flexibility.
