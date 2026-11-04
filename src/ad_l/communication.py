"""普通通信连接校验与拓扑排序，不依赖模型或训练张量。"""

def execution_topology(spatial_edges, temporal_edges, node_ids=("A", "B")):
    """执行侧只接收普通边数据，不接收带梯度的张量。"""
    nodes = tuple(node_ids)
    def validate(edges, spatial):
        normalized = []
        for edge in edges:
            if (not isinstance(edge, (tuple, list)) or len(edge) != 2
                    or any(type(node) is not str or node not in nodes for node in edge)
                    or (spatial and edge[0] == edge[1])):
                raise ValueError("非法执行边或空间自连接")
            pair = tuple(edge)
            if pair in normalized:
                raise ValueError("执行边不能重复")
            normalized.append(pair)
        return normalized
    spatial = validate(spatial_edges, True)
    temporal = validate(temporal_edges, False)
    indegree = {node: sum(target == node for _, target in spatial) for node in nodes}
    pending = [node for node in nodes if indegree[node] == 0]
    order = []
    while pending:
        node = pending.pop(0)
        order.append(node)
        for source, target in spatial:
            if source == node:
                indegree[target] -= 1
                if indegree[target] == 0:
                    pending.append(target)
    if len(order) != len(nodes):
        raise ValueError("空间图必须无环")
    return {"spatial_edges": [list(edge) for edge in spatial],
            "temporal_edges": [list(edge) for edge in temporal], "execution_order": order}


