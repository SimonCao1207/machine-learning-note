# Iterative closest point (ICP)

- ICP is algorithm employed to minimize difference between two point clouds.

- ICP is often used to reconstruct 2D or 3D surfaces from different scans, to localize robots and achieve optimal path planning

- Disadvantage: the algorithm can converage to local optima.

- Dataset used is [Stanford's bunny](https://graphics.stanford.edu/data/3Dscanrep/)

## Runing
```
    pip install -r requirements.txt 
    python main.py
```

## Reference
- [ICP implementation](https://github.com/chengkunli96/ICP/tree/main)
- [FINDING OPTIMAL ROTATION AND TRANSLATION BETWEEN CORRESPONDING 3D POINTS](https://nghiaho.com/?page_id=671)
- [Point Cloud Registration: Beyond the Iterative Closest Point Algorithm](https://www.thinkautonomous.ai/blog/point-cloud-registration/)
