#!/usr/bin/env python3

import ast
import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Set

TORCH_NAMESPACES = {
    'torch', 'torch.nn', 'torch.optim', 'torch.utils', 'torch.utils.data',
    'torch.cuda', 'torch.backends', 'torch.distributed', 'torch.autograd',
    'torch.jit', 'torch.onnx', 'torch.quantization', 'torch.amp',
    'torchvision', 'torchvision.transforms', 'torchvision.datasets',
    'torchvision.models', 'torchvision.utils',
    'torchaudio', 'torchtext',
}

TORCH_API_PATTERNS = {
    'torch.tensor', 'torch.zeros', 'torch.ones', 'torch.randn',
    'torch.cat', 'torch.stack', 'torch.flatten', 'torch.no_grad',
    'torch.save', 'torch.load', 'torch.device', 'torch.cuda.is_available',
    'torch.manual_seed', 'torch.set_grad_enabled',
    'nn.Module', 'nn.Linear', 'nn.Conv2d', 'nn.Conv1d', 'nn.BatchNorm2d',
    'nn.BatchNorm1d', 'nn.LayerNorm', 'nn.Dropout', 'nn.ReLU', 'nn.GELU',
    'nn.Sigmoid', 'nn.Tanh', 'nn.Softmax', 'nn.Embedding',
    'nn.LSTM', 'nn.GRU', 'nn.RNN', 'nn.Transformer',
    'nn.Sequential', 'nn.ModuleList', 'nn.ModuleDict',
    'nn.CrossEntropyLoss', 'nn.MSELoss', 'nn.BCELoss', 'nn.BCEWithLogitsLoss',
    'nn.L1Loss', 'nn.NLLLoss', 'nn.SmoothL1Loss',
    'nn.AdaptiveAvgPool2d', 'nn.AdaptiveMaxPool2d', 'nn.MaxPool2d',
    'nn.AvgPool2d', 'nn.ConvTranspose2d',
    'nn.functional', 'F.relu', 'F.softmax', 'F.cross_entropy',
    'F.log_softmax', 'F.dropout', 'F.interpolate', 'F.pad',
    'nn.utils.rnn.pack_padded_sequence', 'nn.utils.rnn.pad_packed_sequence',
    'nn.utils.clip_grad_norm_', 'nn.DataParallel',
    'optim.SGD', 'optim.Adam', 'optim.AdamW', 'optim.RMSprop',
    'optim.lr_scheduler.StepLR', 'optim.lr_scheduler.CosineAnnealingLR',
    'optim.lr_scheduler.ReduceLROnPlateau', 'optim.lr_scheduler.LambdaScheduler',
    'DataLoader', 'Dataset', 'data.DataLoader', 'data.Dataset',
    'data.TensorDataset', 'data.random_split',
    'transforms.Compose', 'transforms.ToTensor', 'transforms.Normalize',
    'transforms.RandomCrop', 'transforms.RandomHorizontalFlip',
    'transforms.Resize', 'transforms.CenterCrop', 'transforms.ColorJitter',
    'datasets.CIFAR10', 'datasets.MNIST', 'datasets.ImageFolder',
    'datasets.ImageNet',
}

TORCH_METHOD_PATTERNS = {
    'to', 'cuda', 'cpu', 'eval', 'train', 'parameters', 'named_parameters',
    'state_dict', 'load_state_dict', 'zero_grad', 'backward', 'step',
    'item', 'detach', 'clone', 'contiguous', 'view', 'reshape',
    'squeeze', 'unsqueeze', 'permute', 'transpose',
    'max', 'min', 'sum', 'mean', 'size', 'shape', 'numel',
    'requires_grad_', 'no_grad', 'topk', 'argsort',
}

@dataclass
class BoilerplatePattern:
    name: str
    description: str
    category: str
    required_calls: List[str] = field(default_factory=list)
    required_structure: str = ""
    source: str = ""

BOILERPLATE_PATTERNS = [
    BoilerplatePattern(
        name="standard_training_loop",
        description="Standard training loop: zero_grad -> forward -> loss -> backward -> step",
        category="A",
        required_calls=["zero_grad", "backward", "step"],
        required_structure="for_loop_with_backward",
        source="pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
    ),
    BoilerplatePattern(
        name="standard_eval_loop",
        description="Standard evaluation: model.eval() + torch.no_grad()",
        category="A",
        required_calls=["eval", "no_grad"],
        required_structure="eval_with_no_grad",
        source="pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
    ),
    BoilerplatePattern(
        name="nn_module_init",
        description="nn.Module __init__ calling super().__init__()",
        category="A",
        required_calls=["super", "__init__"],
        required_structure="nn_module_init",
        source="pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
    ),
    BoilerplatePattern(
        name="nn_module_forward",
        description="nn.Module forward() method",
        category="A",
        required_calls=[],
        required_structure="nn_module_forward",
        source="pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
    ),
    BoilerplatePattern(
        name="device_setup",
        description="Standard device selection: torch.device('cuda' if ... else 'cpu')",
        category="A",
        required_calls=["torch.device", "cuda.is_available"],
        required_structure="device_setup",
        source="pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html",
    ),
    BoilerplatePattern(
        name="model_checkpoint_save",
        description="Standard torch.save() for model state_dict",
        category="A",
        required_calls=["torch.save", "state_dict"],
        required_structure="checkpoint_save",
        source="pytorch.org/tutorials/beginner/saving_loading_models.html",
    ),
    BoilerplatePattern(
        name="model_checkpoint_load",
        description="Standard torch.load() + load_state_dict()",
        category="A",
        required_calls=["torch.load", "load_state_dict"],
        required_structure="checkpoint_load",
        source="pytorch.org/tutorials/beginner/saving_loading_models.html",
    ),
    BoilerplatePattern(
        name="gradient_clipping",
        description="Standard gradient clipping with clip_grad_norm_",
        category="A",
        required_calls=["clip_grad_norm_"],
        required_structure="",
        source="pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
    ),

    BoilerplatePattern(
        name="dataset_subclass",
        description="Custom Dataset with __len__ and __getitem__",
        category="B",
        required_calls=["__len__", "__getitem__"],
        required_structure="dataset_subclass",
        source="pytorch.org/tutorials/beginner/basics/data_tutorial.html",
    ),
    BoilerplatePattern(
        name="transforms_pipeline",
        description="torchvision.transforms.Compose() pipeline",
        category="B",
        required_calls=["Compose", "ToTensor", "Normalize"],
        required_structure="transforms_compose",
        source="pytorch.org/vision/stable/transforms.html",
    ),
    BoilerplatePattern(
        name="dataloader_setup",
        description="DataLoader instantiation with standard params",
        category="B",
        required_calls=["DataLoader"],
        required_structure="dataloader_setup",
        source="pytorch.org/tutorials/beginner/basics/data_tutorial.html",
    ),
    BoilerplatePattern(
        name="argparse_hyperparams",
        description="argparse for hyperparameter configuration",
        category="B",
        required_calls=["ArgumentParser", "add_argument", "parse_args"],
        required_structure="argparse_pattern",
        source="common ML project pattern",
    ),
]

class ASTAnalyzer:

    def __init__(self):
        self.imports = {}
        self.from_imports = {}

    def analyze_imports(self, tree: ast.Module):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[-1]
                    self.imports[name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    name = alias.asname or alias.name
                    self.from_imports[name] = (module, alias.name)

    def is_torch_import(self, name: str) -> bool:
        if name in self.imports:
            return any(self.imports[name].startswith(ns)
                       for ns in ['torch', 'torchvision', 'torchaudio', 'torchtext'])
        if name in self.from_imports:
            module, _ = self.from_imports[name]
            return any(module.startswith(ns)
                       for ns in ['torch', 'torchvision', 'torchaudio', 'torchtext'])
        return False

    def get_all_calls(self, node: ast.AST) -> List[str]:
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_name = self._resolve_call_name(child.func)
                if call_name:
                    calls.append(call_name)
        return calls

    def _resolve_call_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value_name = self._resolve_call_name(node.value)
            if value_name:
                return f"{value_name}.{node.attr}"
            return node.attr
        return None

    def get_all_attributes(self, node: ast.AST) -> List[str]:
        attrs = []
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                attrs.append(child.attr)
        return attrs

    def count_framework_usage(self, node: ast.AST) -> Tuple[int, int]:
        calls = self.get_all_calls(node)
        attrs = self.get_all_attributes(node)

        framework_count = 0
        total_count = 0

        for call in calls:
            total_count += 1
            parts = call.split('.')
            if any(call.startswith(pat) or call in pat
                   for pat in TORCH_API_PATTERNS):
                framework_count += 1
            elif parts[0] in self.imports and self.is_torch_import(parts[0]):
                framework_count += 1
            elif parts[0] in self.from_imports and self.is_torch_import(parts[0]):
                framework_count += 1
            elif len(parts) >= 2 and parts[-1] in TORCH_METHOD_PATTERNS:
                framework_count += 1

        for attr in attrs:
            if attr in TORCH_METHOD_PATTERNS:
                framework_count += 1
                total_count += 1

        total_statements = 0
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if isinstance(child, ast.stmt):
                    total_statements += 1
        total_count = max(total_count, total_statements)

        return framework_count, max(total_count, 1)

def match_training_loop(node: ast.AST) -> bool:
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)

    has_zero_grad = 'zero_grad' in calls
    has_backward = 'backward' in calls
    has_step = 'step' in calls

    has_for_loop = any(isinstance(child, ast.For) for child in ast.walk(node))

    return has_zero_grad and has_backward and has_step and has_for_loop

def match_eval_pattern(node: ast.AST) -> bool:
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
            elif isinstance(child.func, ast.Name):
                calls.append(child.func.id)

    has_eval = 'eval' in calls
    has_no_grad = any(isinstance(child, ast.With) for child in ast.walk(node))
    for child in ast.walk(node):
        if isinstance(child, ast.With):
            for item in child.items:
                with_call = _get_call_name(item.context_expr)
                if with_call and 'no_grad' in with_call:
                    has_no_grad = True

    return has_eval and has_no_grad

def match_nn_module_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        base_name = _get_full_name(base)
        if base_name and ('Module' in base_name or 'nn.Module' in base_name):
            return True
    return False

def match_dataset_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        base_name = _get_full_name(base)
        if base_name and 'Dataset' in base_name:
            return True
    return False

def match_checkpoint_save(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _get_call_name(child)
            if name and ('torch.save' in name or 'save' in name):
                for arg in child.args:
                    if isinstance(arg, ast.Call):
                        arg_name = _get_call_name(arg)
                        if arg_name and 'state_dict' in arg_name:
                            return True
                    elif isinstance(arg, ast.Dict):
                        return True
    return False

def match_checkpoint_load(node: ast.AST) -> bool:
    calls = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _get_call_name(child)
            if name:
                calls.add(name)

    has_load = any('torch.load' in c or (c == 'load' and 'torch' in str(calls))
                   for c in calls)
    has_load_state = any('load_state_dict' in c for c in calls)
    return has_load and has_load_state

def _get_call_name(node) -> Optional[str]:
    if isinstance(node, ast.Call):
        return _get_full_name(node.func)
    return _get_full_name(node)

def _get_full_name(node) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        value = _get_full_name(node.value)
        if value:
            return f"{value}.{node.attr}"
        return node.attr
    return None

@dataclass
class ClassificationResult:
    file: str
    name: str
    type: str
    line_start: int
    line_end: int
    loc: int
    category: str
    framework_ratio: float
    matched_pattern: str
    confidence: str
    rationale: str
    parent_class: Optional[str] = None

class BoilerplateClassifier:

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self.analyzer = ASTAnalyzer()

    def classify_file(self, filepath: str) -> List[ClassificationResult]:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                source = f.read()
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"  Warning: syntax error in {filepath}: {e}")
            return []

        self.analyzer = ASTAnalyzer()
        self.analyzer.analyze_imports(tree)

        results = []
        rel_path = os.path.basename(filepath)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                result = self._classify_function(node, rel_path)
                results.append(result)

            elif isinstance(node, ast.ClassDef):
                class_result = self._classify_class(node, rel_path)
                results.append(class_result)

                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.FunctionDef):
                        result = self._classify_method(
                            child, rel_path, node.name,
                            is_nn_module=match_nn_module_class(node),
                            is_dataset=match_dataset_class(node),
                        )
                        results.append(result)

        return results

    def _classify_function(self, node: ast.FunctionDef,
                           filepath: str) -> ClassificationResult:
        fw_count, total = self.analyzer.count_framework_usage(node)
        ratio = fw_count / max(total, 1)
        loc = node.end_lineno - node.lineno + 1 if node.end_lineno else 1

        if match_training_loop(node):
            return ClassificationResult(
                file=filepath, name=node.name, type='function',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='A', framework_ratio=ratio,
                matched_pattern='standard_training_loop',
                confidence='high',
                rationale='Contains training loop pattern: zero_grad -> forward -> backward -> step',
            )

        if match_eval_pattern(node):
            return ClassificationResult(
                file=filepath, name=node.name, type='function',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='A', framework_ratio=ratio,
                matched_pattern='standard_eval_loop',
                confidence='high',
                rationale='Contains eval pattern: model.eval() + torch.no_grad()',
            )

        if match_checkpoint_save(node):
            return ClassificationResult(
                file=filepath, name=node.name, type='function',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='A', framework_ratio=ratio,
                matched_pattern='model_checkpoint_save',
                confidence='high',
                rationale='Contains standard torch.save(state_dict) pattern',
            )

        if match_checkpoint_load(node):
            return ClassificationResult(
                file=filepath, name=node.name, type='function',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='A', framework_ratio=ratio,
                matched_pattern='model_checkpoint_load',
                confidence='high',
                rationale='Contains standard torch.load + load_state_dict pattern',
            )

        return self._classify_by_ratio(
            node, filepath, 'function', ratio, loc)

    def _classify_method(self, node: ast.FunctionDef, filepath: str,
                         parent_class: str, is_nn_module: bool = False,
                         is_dataset: bool = False) -> ClassificationResult:
        fw_count, total = self.analyzer.count_framework_usage(node)
        ratio = fw_count / max(total, 1)
        loc = node.end_lineno - node.lineno + 1 if node.end_lineno else 1

        if is_nn_module and node.name == '__init__':
            return ClassificationResult(
                file=filepath, name=f"{parent_class}.{node.name}",
                type='method',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='A', framework_ratio=ratio,
                matched_pattern='nn_module_init',
                confidence='high',
                rationale='nn.Module __init__ — framework-required constructor pattern',
                parent_class=parent_class,
            )

        if is_nn_module and node.name == 'forward':
            return ClassificationResult(
                file=filepath, name=f"{parent_class}.{node.name}",
                type='method',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='A', framework_ratio=ratio,
                matched_pattern='nn_module_forward',
                confidence='high',
                rationale='nn.Module forward() — framework-required method',
                parent_class=parent_class,
            )

        if is_dataset and node.name in ('__len__', '__getitem__'):
            return ClassificationResult(
                file=filepath, name=f"{parent_class}.{node.name}",
                type='method',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='B', framework_ratio=ratio,
                matched_pattern='dataset_subclass',
                confidence='high',
                rationale=f'Dataset required method: {node.name}',
                parent_class=parent_class,
            )

        if match_training_loop(node):
            return ClassificationResult(
                file=filepath, name=f"{parent_class}.{node.name}",
                type='method',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='A', framework_ratio=ratio,
                matched_pattern='standard_training_loop',
                confidence='high',
                rationale='Training loop pattern in method',
                parent_class=parent_class,
            )

        result = self._classify_by_ratio(
            node, filepath, 'method', ratio, loc)
        result.name = f"{parent_class}.{node.name}"
        result.parent_class = parent_class
        return result

    def _classify_class(self, node: ast.ClassDef,
                        filepath: str) -> ClassificationResult:
        loc = node.end_lineno - node.lineno + 1 if node.end_lineno else 1

        if match_nn_module_class(node):
            return ClassificationResult(
                file=filepath, name=node.name, type='class',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='A', framework_ratio=1.0,
                matched_pattern='nn_module_class',
                confidence='high',
                rationale='Class inherits from nn.Module — framework-dictated structure',
            )

        if match_dataset_class(node):
            return ClassificationResult(
                file=filepath, name=node.name, type='class',
                line_start=node.lineno, line_end=node.end_lineno or node.lineno,
                loc=loc, category='B', framework_ratio=0.8,
                matched_pattern='dataset_subclass',
                confidence='high',
                rationale='Class inherits from Dataset — API protocol',
            )

        return ClassificationResult(
            file=filepath, name=node.name, type='class',
            line_start=node.lineno, line_end=node.end_lineno or node.lineno,
            loc=loc, category='C', framework_ratio=0.0,
            matched_pattern='none',
            confidence='medium',
            rationale='Class does not inherit from PyTorch base classes',
        )

    def _classify_by_ratio(self, node: ast.FunctionDef, filepath: str,
                           unit_type: str, ratio: float,
                           loc: int) -> ClassificationResult:
        if ratio >= self.threshold:
            category = 'B'
            confidence = 'medium'
            rationale = (f'Framework ratio {ratio:.2f} >= threshold {self.threshold} '
                         f'but no structural pattern match — classified as API protocol')
        elif ratio >= self.threshold * 0.5:
            category = 'B'
            confidence = 'low'
            rationale = (f'Framework ratio {ratio:.2f} is moderate — '
                         f'likely API protocol code')
        else:
            category = 'C'
            confidence = 'high' if ratio < 0.2 else 'medium'
            rationale = (f'Framework ratio {ratio:.2f} < threshold {self.threshold} '
                         f'— classified as custom implementation')

        return ClassificationResult(
            file=filepath, name=node.name, type=unit_type,
            line_start=node.lineno, line_end=node.end_lineno or node.lineno,
            loc=loc, category=category, framework_ratio=ratio,
            matched_pattern='none',
            confidence=confidence,
            rationale=rationale,
        )

EXCLUDE_PATTERNS = [
    'test', 'tests', 'test_', '_test.py', 'conftest',
    'setup.py', 'setup.cfg', '__pycache__', '.git',
    'docs', 'doc', 'venv', 'env', '.tox',
]

def classify_project(project_dir: str, threshold: float = 0.70) -> dict:
    classifier = BoilerplateClassifier(threshold=threshold)
    project_name = os.path.basename(project_dir.rstrip('/'))

    all_results = []
    files_analyzed = 0

    for py_file in sorted(Path(project_dir).rglob('*.py')):
        path_str = str(py_file).lower()
        if any(excl in path_str for excl in EXCLUDE_PATTERNS):
            continue

        rel_path = str(py_file.relative_to(project_dir))
        results = classifier.classify_file(str(py_file))

        for r in results:
            r.file = rel_path

        all_results.extend(results)
        files_analyzed += 1

    manifest = {
        'project': project_name,
        'project_dir': str(project_dir),
        'threshold': threshold,
        'files_analyzed': files_analyzed,
        'total_units': len(all_results),
        'summary': {
            'category_A': sum(1 for r in all_results if r.category == 'A'),
            'category_B': sum(1 for r in all_results if r.category == 'B'),
            'category_C': sum(1 for r in all_results if r.category == 'C'),
            'loc_A': sum(r.loc for r in all_results if r.category == 'A'),
            'loc_B': sum(r.loc for r in all_results if r.category == 'B'),
            'loc_C': sum(r.loc for r in all_results if r.category == 'C'),
        },
        'classifications': [asdict(r) for r in all_results],
    }

    total = manifest['summary']['category_A'] + manifest['summary']['category_B'] + manifest['summary']['category_C']
    if total > 0:
        manifest['summary']['pct_A'] = round(100.0 * manifest['summary']['category_A'] / total, 1)
        manifest['summary']['pct_B'] = round(100.0 * manifest['summary']['category_B'] / total, 1)
        manifest['summary']['pct_C'] = round(100.0 * manifest['summary']['category_C'] / total, 1)
    else:
        manifest['summary']['pct_A'] = 0
        manifest['summary']['pct_B'] = 0
        manifest['summary']['pct_C'] = 0

    return manifest

def print_summary(manifest: dict):
    s = manifest['summary']
    total = s['category_A'] + s['category_B'] + s['category_C']
    total_loc = s['loc_A'] + s['loc_B'] + s['loc_C']

    print(f"\n{'=' * 60}")
    print(f"PROJECT: {manifest['project']}")
    print(f"{'=' * 60}")
    print(f"Files analyzed:     {manifest['files_analyzed']}")
    print(f"Code units found:   {total}")
    print(f"Total LOC:          {total_loc}")
    print(f"\nClassification (by count):")
    print(f"  Category A (Boilerplate):   {s['category_A']:3d} ({s['pct_A']:5.1f}%)")
    print(f"  Category B (API Protocol):  {s['category_B']:3d} ({s['pct_B']:5.1f}%)")
    print(f"  Category C (Custom):        {s['category_C']:3d} ({s['pct_C']:5.1f}%)")
    print(f"\nClassification (by LOC):")
    if total_loc > 0:
        print(f"  Category A: {s['loc_A']:5d} LOC ({100.0*s['loc_A']/total_loc:5.1f}%)")
        print(f"  Category B: {s['loc_B']:5d} LOC ({100.0*s['loc_B']/total_loc:5.1f}%)")
        print(f"  Category C: {s['loc_C']:5d} LOC ({100.0*s['loc_C']/total_loc:5.1f}%)")

    print(f"\nDetailed classifications:")
    for c in manifest['classifications']:
        marker = {'A': 'BOILER', 'B': 'PROTO ', 'C': 'CUSTOM'}[c['category']]
        print(f"  [{marker}] {c['file']}:{c['name']} "
              f"(L{c['line_start']}-{c['line_end']}, "
              f"fw_ratio={c['framework_ratio']:.2f}, "
              f"pattern={c['matched_pattern']}, "
              f"conf={c['confidence']})")

def main():
    parser = argparse.ArgumentParser(
        description="AST-based boilerplate classifier for PyTorch projects"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--repo', help='Single project directory to classify')
    group.add_argument('--repos', help='Directory containing multiple project directories')
    parser.add_argument('--output', default=None,
                        help='Output directory for JSON manifests')
    parser.add_argument('--threshold', type=float, default=0.70,
                        help='Framework ratio threshold (default: 0.70)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    args = parser.parse_args()

    if args.repo:
        projects = [args.repo]
    else:
        projects = sorted([
            os.path.join(args.repos, d)
            for d in os.listdir(args.repos)
            if os.path.isdir(os.path.join(args.repos, d))
            and not d.startswith('.')
        ])

    if args.output:
        os.makedirs(args.output, exist_ok=True)

    all_manifests = []
    for project_dir in projects:
        print(f"\nClassifying: {project_dir}")
        manifest = classify_project(project_dir, threshold=args.threshold)
        all_manifests.append(manifest)

        if not args.quiet:
            print_summary(manifest)

        if args.output:
            out_file = os.path.join(args.output,
                                    f"{manifest['project']}_manifest.json")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            print(f"  Manifest saved to: {out_file}")

    if len(all_manifests) > 1:
        total_a = sum(m['summary']['category_A'] for m in all_manifests)
        total_b = sum(m['summary']['category_B'] for m in all_manifests)
        total_c = sum(m['summary']['category_C'] for m in all_manifests)
        total = total_a + total_b + total_c

        print(f"\n{'=' * 60}")
        print(f"AGGREGATE ACROSS {len(all_manifests)} PROJECTS")
        print(f"{'=' * 60}")
        print(f"Total code units: {total}")
        if total > 0:
            print(f"  Category A: {total_a} ({100.0*total_a/total:.1f}%)")
            print(f"  Category B: {total_b} ({100.0*total_b/total:.1f}%)")
            print(f"  Category C: {total_c} ({100.0*total_c/total:.1f}%)")

if __name__ == '__main__':
    main()
