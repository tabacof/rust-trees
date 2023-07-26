# Decisision Tree Classifier


# Scikit Learn implementation

Psuedo code of the Sklearn implementation

## DecisionTreeRegressor


```python
class DecisionTreeRegressor:
    def fit(X, y):
    	# a lot of checks

    	# Build tree
    	criterion = criterion.MSE(n_outputs, n_samples)
    	
    	splitter = splitter.BestSplitter(criterion)	
	
	tree = Tree(n_features, n_outputs)
	
	builder = DepthFirstTreeBuilder(splitter)
	
	builder.build(tree, X, y)
	
	prune_tree(tree)

```


## DepthFirstTreeBuilder


```python

cdef class DepthFirstTreeBuilder(TreeBuilder):
    cpdef build(tree, X, y):
        splitter.init(X, y)

        stack.push({start: 0, end: n_node_samples, depth: 0, 
            parent: None, is_left: False})

        while not stack.empty():
            sr = stack.pop()

            if check_is_leaf(sr):
                tree._add_leaf_node(sr)
                continue

            split = splitter.node_split(sr.start, sr.end)
            node_id = tree._add_node(sr, split)

            stack.push({start: split.pos, end: sr.end, depth: sr.depth + 1,
                parent: node_id, is_left: False})

            stack.push({start: sr.start, end: split.pos, depth: sr.depth + 1,
                parent: node_id, is_left: True})
        # return

   
```

## BestSplitter


```python
cdef class BestSplitter(Splitter):
    cdef int init(X, criterion):
        self.partitioner = DensePartitioner(X)
    
    cdef int node_split(self, int start, int end) except -1 nogil:
        best_split = _init_split(end) # null-like split
        partitioner.init_node_split(start, end)
    
        for f in range(n_features):
            criterion.reset()
            partitioner.sort_samples_and_feature_values(f)
    
            p_prev, p = partitioner.next_pos(start) 
            while p < end: # Evaluate all splits
                criterion.update(p)

                imp = criterion.proxy_impurity_improvement()
                if imp > best_imp:
                    best_imp = imp
                    threshold = (X[p_prev, f] + X[p, f]) / 2.0
                    best_split = Split(pos, threshold, f) 

                p_prev, p = partitioner.next_pos(p)
    
        # Reorganize into samples[start:best_split.pos] + samples[best_split.pos:end]
        partitioner.partition_samples_final(best_split.pos, best_split.threshold, best_split.feature)
    
        return best_split

```

## Still left to study

### Tree

```python
cdef class Tree:
    """Array-based representation of a binary decision tree."""
    ...
```

### DensePartitioner

```python
cdef class DensePartitioner:
    """Partitioner for dense data."""
```

### MSE
```python
cdef class MSE(Criterion):
    """Mean squared error criterion."""
```

