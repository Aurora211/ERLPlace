import treelib
import numpy as np

class SolutionPool:
    def __init__(self, pool_size: int, alpha: float) -> None:
        self.pool_size = pool_size
        self.alpha = alpha
        self.best_hpwl = 1e9
        self.best_hpwl_max = 1e9
        self.sols = []
        self.tree = treelib.Tree()
        self.tree.create_node(
            tag="root",
            identifier=str([]),
            data={
                "visit_time": 0,
                "hpwl": 1e9,
                "hist_hpwl": {},
                "score": 0.0
            }
        )
        self.frontiers = [[]]
        self.frontiers_hist = [[]]
    @property
    def best_solution(self):
        return self.sols[0]["placement"], self.sols[0]["solution"], self.sols[0]["hpwl"]
    @property
    def depth_min_max(self):
        depth = [len(x) for x in self.frontiers]
        return min(depth), max(depth)
    def update_solution(self, frontier_id, solution, hpwl, placement):
        if hpwl < self.best_hpwl_max:
            sol_record = {
                "solution": solution,
                "placement": placement,
                "hpwl": hpwl,
            }
            self.sols.append(sol_record)
            self.sols = sorted(self.sols, key=lambda x: x["hpwl"])
            self.sols = self.sols[:self.pool_size]
            self.best_hpwl_max = self.sols[-1]["hpwl"]
            self.best_hpwl = self.sols[0]["hpwl"]
        for i in range(len(solution)):
            node = self.tree.get_node(str(solution[:i]))
            if node:
                node.data["visit_time"] += 1
                node.data["score"] = - node.data["hpwl"] + self.alpha * 1 / np.sqrt(node.data["visit_time"])
                if i < len(solution) - 1 and solution[:i+1] not in self.frontiers_hist:
                    node.data["hist_hpwl"][str(solution[i])] = hpwl.item() if str(
                        solution[i]) not in node.data["hist_hpwl"] else min(node.data["hist_hpwl"][str(solution[i])], hpwl.item())
                    node.data["hpwl"] = min(node.data["hpwl"], hpwl.item())
            if node is None:
                self.tree.create_node(
                    tag=str(solution[i-1]),
                    identifier=str(solution[:i]),
                    parent=str(solution[:i-1]),
                    data={
                        "visit_time": 1,
                        "hpwl": hpwl.item(),
                        "score": - hpwl.item() + self.alpha * 1 / np.sqrt(1),
                        "hist_hpwl": {str(solution[i]): hpwl.item()},
                    }
                )
    def update_frontiers(self):
        front_scores = [(self.tree.get_node(str(x)).data["score"], x)
                        for x in self.frontiers]
        front_scores = sorted(
            front_scores, key=lambda x: self.tree.get_node(str(x[1])).data["hpwl"])
        if len(front_scores) > self.pool_size:
            raise ValueError("Frontier size is larger than pool size")
        self.frontiers_cand = []
        for front in self.frontiers:
            for chid in self.tree.children(str(front)):
                if eval(chid.identifier) not in self.frontiers and eval(chid.identifier) not in self.frontiers_cand:
                    self.frontiers_cand.append(eval(chid.identifier))
        cand_scores = [(self.tree.get_node(str(x)).data["score"], x) for x in self.frontiers_cand]
        cand_scores = sorted(
            cand_scores, key=lambda x: self.tree.get_node(str(x[1])).data["hpwl"])
        while len(cand_scores) > 0 and (cand_scores[0][0] >= front_scores[-1][0] or len(front_scores) < self.pool_size):
            node = self.tree.get_node(str(cand_scores[0][1]))
            parent = self.tree.parent(node.identifier)
            if node.tag in parent.data["hist_hpwl"]:
                parent.data["hist_hpwl"].pop(node.tag)
            parent.data["hpwl"] = 1e9 if len(parent.data["hist_hpwl"].values()) == 0 else min(parent.data["hist_hpwl"].values())
            parent.data["score"] = - parent.data["hpwl"] + self.alpha * 1 / np.sqrt(parent.data["visit_time"])
            front_scores.append(cand_scores[0])
            front_scores = sorted(front_scores, key=lambda x: x[0], reverse=True)
            self.frontiers.append(cand_scores[0][1])
            self.frontiers_hist.append(cand_scores[0][1])
            if len(self.frontiers) > self.pool_size:
                self.frontiers.remove(front_scores[-1][1])
                front_scores = front_scores[:-1]
            self.frontiers_cand.remove(cand_scores[0][1])
            cand_scores = cand_scores[1:]
    def sample(self):
        frontier_id = np.random.choice(len(self.frontiers))
        return frontier_id, self.frontiers[frontier_id]


