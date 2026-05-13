# We implement the evaluation metric in this file.
from rdkit import Chem
from torch_geometric.data import Data
from core.evaluation.utils import scoring_func

from core.evaluation.utils import (
    check_stability,
    convert_atomcloud_to_mol_smiles,
    mol2smiles,
)
from core.evaluation.docking_qvina import QVinaDockingTask
from core.evaluation.docking_vina import VinaDockingTask
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
import os
from posecheck import PoseCheck
from copy import deepcopy
from core.evaluation.basic_results import BasicResults
import multiprocessing as mp
from multiprocessing import Pool, Manager
import pickle
import traceback
# from redock_pt_results import redock


class CondMolGenMetric(object):
    def __init__(
        self, atom_decoder, atom_enc_mode, type_one_hot, single_bond, docking_config
    ):
        self.atom_decoder = atom_decoder
        self.atom_enc_mode = atom_enc_mode
        self.type_one_hot = type_one_hot
        self.single_bond = single_bond
        self.docking_config = docking_config
        
    def _process_chunk(self, chunk_data):
        """处理一个数据块，用于多进程"""
        chunk_idx, generated_chunk, atom_decoder, single_bond, docking_config = chunk_data
        results_chunk = []
        
        for item in generated_chunk:
            item_result = deepcopy(item)
            
            # 计算稳定性
            try:
                positions = item['pred_pos']
                atom_type = item['pred_v']
                stability_results = check_stability(
                    positions=positions,
                    atom_type=atom_type,
                    single_bond=single_bond,
                )
                item_result['stability'] = {
                    "mol_stable": int(stability_results[0]),
                    "nr_stable_bonds": int(stability_results[1]),
                    "n_atoms": int(stability_results[2])
                }
            except Exception as e:
                item_result['stability'] = None
                item_result['stability_error'] = str(e)
            
            # 计算化学性质
            try:
                mol = item['mol']
                chem_results = scoring_func.get_chem(mol)
                chem_results['atom_num'] = mol.GetNumAtoms()
                item_result['chem_results'] = chem_results
            except Exception as e:
                item_result['chem_results'] = None
                item_result['chem_error'] = str(e)
            
            # 分子对接
            try:
                if docking_config is not None:
                    mol = item['mol']
                    ligand_filename = item['ligand_filename']
                    pos = item['pred_pos']
                    
                    if docking_config.mode in ['vina_score', 'vina_dock']:
                        vina_task = VinaDockingTask.from_generated_mol(
                            mol, ligand_filename, pos=pos, protein_root=docking_config.protein_root)
                        score_only_results = vina_task.run(mode='score_only', exhaustiveness=docking_config.exhaustiveness)
                        minimize_results = vina_task.run(mode='minimize', exhaustiveness=docking_config.exhaustiveness)
                        vina_results = {
                            'score_only': score_only_results,
                            'minimize': minimize_results,
                        }
                        if docking_config.mode == 'vina_dock':
                            docking_results = vina_task.run(mode='dock', exhaustiveness=docking_config.exhaustiveness)
                            vina_results['dock'] = docking_results
                        item_result['vina'] = vina_results
            except Exception as e:
                item_result['vina'] = None
                item_result['vina_error'] = str(e)
            
            # PoseCheck分析
            try:
                pc = PoseCheck()
                pc.load_ligands_from_mols([item['mol']])
                strain = pc.calculate_strain_energy()[0]
                item_result['pose_check'] = {
                    'strain': strain,
                }
            except Exception as e:
                item_result['pose_check'] = None
                item_result['pose_check_error'] = str(e)
            
            results_chunk.append(item_result)
        
        return chunk_idx, results_chunk

    def compute_stability(self, generated: list[dict]):
        n_samples = len(generated)
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        for data in generated:
            positions = data['pred_pos']
            atom_type = data['pred_v']
            
            stability_results = check_stability(
                positions=positions,
                atom_type=atom_type,
                single_bond=self.single_bond,
            )
            
            molecule_stable += int(stability_results[0])
            nr_stable_bonds += int(stability_results[1])
            n_atoms += int(stability_results[2])

        # stability
        fraction_mol_stable = molecule_stable / float(n_samples)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        stability_dict = {
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
        }
        return stability_dict

    def compute_chem_results(self, generated: list[dict]):
        """串行版本，保留原逻辑"""
        pc = PoseCheck()
        last_protein_fn = None

        for item in tqdm(generated, total=len(generated), desc="Chem eval"):
            mol = item['mol']

            try:
                ligand_filename = item['ligand_filename']
                pos = item['pred_pos']

                # qed, logp, sa, lipinski, etc
                chem_results = scoring_func.get_chem(mol)
                chem_results['atom_num'] = mol.GetNumAtoms()
                item['chem_results'] = chem_results
            except Exception as e:
                print(f'[CHEM FAIL] {e}')
            
            try:
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                complete = smiles is not None and '.' not in smiles
                validity = smiles is not None
                 
                item['complete'] = complete
                item['validity'] = validity
            except Exception as e:
                print(f'[VALIDITY FAIL] {e}')

            try:
                # docking
                if self.docking_config is not None:
                    if self.docking_config.mode == 'qvina':
                        raise NotImplementedError("QVina is not supported in this version.")
                    elif self.docking_config.mode in ['vina_score', 'vina_dock']:
                        vina_task = VinaDockingTask.from_generated_mol(
                            mol, ligand_filename, pos=pos, protein_root=self.docking_config.protein_root)
                        score_only_results = vina_task.run(mode='score_only', exhaustiveness=self.docking_config.exhaustiveness)
                        minimize_results = vina_task.run(mode='minimize', exhaustiveness=self.docking_config.exhaustiveness)
                        vina_results = {
                            'score_only': score_only_results,
                            'minimize': minimize_results,
                        }
                        if self.docking_config.mode == 'vina_dock':
                            docking_results = vina_task.run(mode='dock', exhaustiveness=self.docking_config.exhaustiveness)
                            vina_results['dock'] = docking_results
                        item['vina'] = vina_results
                    else:
                        raise NotImplementedError(f"Unknown docking mode: {self.docking_config.mode}")
            except Exception as e:
                print(f'[VINA FAIL] {e}')

            try:
                pc.load_ligands_from_mols([mol])
                strain = pc.calculate_strain_energy()[0]
                item['pose_check'] = {
                    'strain': strain,
                }
            except Exception as e:
                print(f'[POSE CHECK FAIL] {e}')
                
    def compute_chem_results_parallel(self, generated: list[dict], num_processes: int = None):
        """并行计算化学性质、对接和构象分析"""
        if num_processes is None:
            num_processes = max(1, mp.cpu_count() - 1)  # 留一个CPU核心
        
        # 分割数据为块
        chunk_size = max(1, len(generated) // num_processes)
        chunks = []
        for i in range(0, len(generated), chunk_size):
            chunks.append((i // chunk_size, generated[i:i+chunk_size], 
                          self.atom_decoder, self.single_bond, self.docking_config))
        
        print(f"使用 {num_processes} 个进程处理 {len(generated)} 个分子，分块大小: {chunk_size}")
        
        # 使用进程池并行处理
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(self._process_chunk, chunks), 
                               total=len(chunks), 
                               desc="并行评估"))
        
        # 按原始顺序重新组合结果
        results.sort(key=lambda x: x[0])  # 按chunk_idx排序
        all_results = []
        for _, chunk_results in results:
            all_results.extend(chunk_results)
        
        return all_results

    def evaluate(self, generated: list[dict], bad_case_dir: str = None, num_processes: int = None):
        """generated: list of pairs 
        (positions: n x 3, atom_types: n x K [int] if type_one_hot else n [int])
        the positions and atom types should already be masked."""

        print(f"开始评估 {len(generated)} 个分子...")
        
        # 方法1: 并行计算所有指标
        if num_processes is not None and num_processes > 1:
            print("使用并行模式...")
            generated = self.compute_chem_results_parallel(generated, num_processes)
        else:
            print("使用串行模式...")
            self.compute_chem_results(generated)  # 串行版本
        
        # 计算稳定性（需要单独计算）
        stability_dict = self.compute_stability(generated)

        # 使用BasicResults进行统计分析
        results = BasicResults('bfn', 'molcraft', generated)

        def stat1(arr, name):
            n_total = len(arr)
            isnan = np.isnan(arr)
            n_isnan = isnan.sum()
            arr2 = arr[~isnan]
            return {
                f'{name}_fail': n_isnan / n_total,
                f'{name}_mean': np.mean(arr2)
            }
        
        metrics = {**stability_dict}
        metrics.update(stat1(results.qed_list, 'qed'))
        metrics.update(stat1(results.sa_list, 'sa'))

        def save_bad_case(idx, res):
            if bad_case_dir is None: 
                return
            os.makedirs(bad_case_dir, exist_ok=True)
            mol = res['mol']
            ligand_filename = res["ligand_filename"]
            atom_num = res['chem_results']['atom_num']
            center_change = res.get('center_change', np.nan)
            mol_pos_range = res.get('mol_pos_range', np.nan)
            qed = res['chem_results']['qed']
            sa = res['chem_results']['sa']
            lipinski = res['chem_results']['lipinski']
            
            if 'vina' in res and res['vina'] is not None:
                vina_score = res['vina']['score_only'][0]['affinity']
                vina_min = res['vina']['minimize'][0]['affinity']
            else:
                vina_score = np.nan
                vina_min = np.nan
            
            if 'pose_check' in res and res['pose_check'] is not None:
                strain = res['pose_check']['strain']
            else:
                strain = np.nan
            
            mol.SetProp('_Name', ligand_filename)
            mol.SetProp('atom_num', str(atom_num))
            mol.SetProp('center_change', str(center_change))
            mol.SetProp('mol_pos_range', str(mol_pos_range))
            mol.SetProp('qed', str(qed))
            mol.SetProp('sa', str(sa))
            mol.SetProp('lipinski', str(lipinski))
            mol.SetProp('vina_score', str(vina_score))
            mol.SetProp('vina_min', str(vina_min))
            mol.SetProp('strain', str(strain))
            
            with Chem.SDWriter(os.path.join(bad_case_dir, f'{idx}.sdf')) as w:
                w.write(mol)

        pos_vina_msg = {}
        no_vina_msg = {}
        for idx, res in enumerate(results):
            ligand_filename = res["ligand_filename"]
            try:
                if 'vina' in res and res['vina'] is not None:
                    vina_score = res['vina']['score_only'][0]['affinity']
                    vina_min = res['vina']['minimize'][0]['affinity']
                    if vina_score > 0 or vina_min > 0:
                        if ligand_filename not in pos_vina_msg:
                            pos_vina_msg[ligand_filename] = ''
                        
                        _ = deepcopy(res)
                        del _['pred_pos'], _['pred_v'], _['is_aromatic'], _['mol'], _['protein_center'], _['mol_center']
                        _['vina'] = {
                            'vina_score': vina_score,
                            'vina_minimize': vina_min,
                        }
                        
                        pos_vina_msg[ligand_filename] += f'{idx} {_}\n\n'
                        save_bad_case(idx, res)
            except Exception as e:
                if ligand_filename not in no_vina_msg:
                    no_vina_msg[ligand_filename] = []
                no_vina_msg[ligand_filename].append(idx)
        
        if len(pos_vina_msg):
            for k, v in pos_vina_msg.items():
                print(f'[POS VINA] ligand_fn = {k}, n_ligand = {len(v)}')
                print(f'[POS VINA] ligand index = {v}')
        if len(no_vina_msg):
            for k, v in no_vina_msg.items():
                print(f'[NO VINA] ligand_fn = {k}, n_ligand = {len(v)}')
                print(f'[NO VINA] ligand index = {v}')                

        def stat2(arr, name):
            n_total = len(arr)
            isnan = np.isnan(arr)
            n_isnan = isnan.sum()
            arr2 = arr[~isnan]
            return {
                f'{name}_fail': n_isnan / n_total,
                f'{name}_mean': np.mean(arr2),
                f'{name}_median': np.median(arr2),
                f'{name}_neg_mean': np.mean(arr2[arr2 < 0]),
                f'{name}_neg_ratio': (arr2 < 0).sum() / len(arr2),
            }

        if 'vina' in results[0] and results[0]['vina'] is not None:
            vina_score_list = results.vina_score_list
            metrics.update(stat2(vina_score_list, 'vina_score'))
            if 'minimize' in results[0]['vina']:
                vina_min_list = results.vina_min_list
                metrics.update(stat2(vina_min_list, 'vina_minimize'))
            if 'dock' in results[0]['vina']:
                vina_dock_list = results.vina_dock_list
                metrics.update(stat2(vina_dock_list, 'vina_dock'))

        def stat3(arr, name):
            n_total = len(arr)
            isnan = np.isnan(arr)
            n_isnan = isnan.sum()
            arr2 = arr[~isnan]
            perc = np.percentile(arr2, [25, 50, 75])
            return {
                f'{name}_fail': n_isnan / n_total,
                f'{name}_25': perc[0],
                f'{name}_50': perc[1],
                f'{name}_75': perc[2],
            }

        metrics.update(stat3(results.strain_list, 'strain'))
        
        if 'validity' in results[0]:
            metrics.update({'validity':sum(results.validity_list)/len(results.validity_list)})
            metrics.update({'completeness':sum(results.complete_list)/len(results.complete_list)})
    
        return metrics


# 添加一个简单的并行评估函数作为入口点
def evaluate_parallel(generated_list, atom_decoder, single_bond, docking_config, 
                     bad_case_dir=None, num_processes=None):
    """
    快速并行评估入口函数
    """
    metric = CondMolGenMetric(
        atom_decoder=atom_decoder,
        atom_enc_mode=None,
        type_one_hot=None,
        single_bond=single_bond,
        docking_config=docking_config
    )
    
    return metric.evaluate(generated_list, bad_case_dir, num_processes)