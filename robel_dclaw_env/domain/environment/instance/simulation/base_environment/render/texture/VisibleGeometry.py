


class VisibleGeometry:
    def __init__(self, sim, task_relevant_geom_group_name: str):
        self.model                         = sim.model
        self.task_relevant_geom_group_name = task_relevant_geom_group_name
        self.visible_geom_group            = [0, 1, 2] # XMLファイル内での group 番号と整合性が取れるように設定する


    def get_task_relevant_grouped_visible_geometries(self):
        all_visible_geoms           = self._get_all_visible_geoms()
        task_relevant_visible_geoms = self._get_task_relevant_visible_geoms(all_visible_geoms)
        return self._grouping_task_relevant_visible_geoms(
            all_visible_geoms, task_relevant_visible_geoms
        )


    def _get_all_visible_geoms(self):
        visible_geom = []
        for name in self.model.geom_names:
            id    = self.model.geom_name2id(name)
            group = self.model.geom_group[id]
            if group in self.visible_geom_group:
                visible_geom.append(name)
        return visible_geom


    def _get_task_relevant_visible_geoms(self, visible_geoms):
        return [x for x in visible_geoms if self.task_relevant_geom_group_name in x]


    def _grouping_task_relevant_visible_geoms(self,
            all_visible_geoms          : list,
            task_relevant_visible_geoms: list
        ):
        [all_visible_geoms.remove(name) for name in task_relevant_visible_geoms] # 抽出したgeomをもとのlistから削除

        for i in range(len(all_visible_geoms)):
            # タスク関連のgeomとデータ形式を合わせるために元の各要素をlistでグループ化する
            all_visible_geoms[i] = [all_visible_geoms[i]] # list() 関数だと1文字ずつ分解されるのでダメ

        all_visible_geoms.append(task_relevant_visible_geoms) # タスク関連geomをその他のgeomと結合させる
        return all_visible_geoms
