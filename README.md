MIMO-LP: A Multi-Input Multi-Output Framework for Subgraph-based Link Prediction
===============================================================================
Introduction

MIMOLP identifies that existing subgraph-based link prediction methods suffer from substantial redundant message passing in overlapping regions between two subgraphs. However, such redundancy cannot be directly eliminated due to distinct contextual features across subgraphs. To address this issue, we superimpose cross-subgraph contextual features from an orthogonal perspective, thereby eliminating redundant message passing and accelerating link prediction.

===============================================================================

Install Packages 

pip install -r requirements.txt


===============================================================================

Quick Start

Type:  

python Main.py  --model-name M-SEAL --data-name  NS   --multiplexing-count 40 --test-ratio  0.2

to test dataset NS with multiplexing count 40 on backbone M-SEAL.

Type:  

python Main.py  --model-name M-SEAL --data-name  Yeast   --multiplexing-count 50 --test-ratio  0.2

to test dataset Yeast with multiplexing count 50 on backbone M-SEAL.

Type:  

python Main.py  --model-name M-SEAL --data-name  Drugbank   --multiplexing-count 50 --test-ratio  0.2

to test dataset Drugbank with multiplexing count 50 on backbone M-SEAL.

Type:  

python Main.py  --model-name M-PS2 --data-name  NS   --multiplexing-count 40 --test-ratio  0.2

to test dataset NS with multiplexing count 40 on backbone M-PS2.

Type:  

python Main.py  --model-name M-PS2 --data-name  Yeast   --multiplexing-count 50 --test-ratio  0.2

to test dataset Yeast with multiplexing count 50 on backbone M-PS2.

Type:  

python Main.py  --model-name M-PS2 --data-name  Drugbank   --multiplexing-count 50 --test-ratio  0.2

to test dataset Drugbank with multiplexing count 50 on backbone M-PS2.

