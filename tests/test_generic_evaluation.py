"""Synthetic checks; all generated diagnostics are written to temporary folders."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src import batch_correction_visualization as vis
from src.pipeline import apply_correction


MARKERS = ['M0', 'M1', 'M2', 'M3']


def frames(n=120):
    rng = np.random.default_rng(73)
    before = pd.DataFrame(rng.lognormal(size=(n, 4)), columns=MARKERS)
    before['M0'] = np.arange(n, dtype=float) + 1
    before['event_id'] = np.arange(n)
    before['Time'] = np.arange(n)
    before['plate'] = np.repeat(['P1', 'P2'], n // 2)
    before['celltypes'] = np.tile(['T', 'B'], n // 2)
    before['sample_name'] = 'anchor'
    before['original_filename'] = before['plate'] + '_anchor.csv'
    before['dataset'] = 'Synthetic'
    before['panel'] = 'Example'
    before['condition'] = 'control'
    after = before.copy(deep=True)
    after['M1'] = after['M1'] * 0.9 + 0.2
    return before, after


def save_small(path, fig=None, **kwargs):
    (fig or vis.plt.gcf()).savefig(path, dpi=40)


class PairingTests(unittest.TestCase):
    def test_matching_events_and_metadata(self):
        before, after = frames()
        after['plate'] = after['plate'].astype('category')
        vis.validate_paired_data(before, after, label_key='celltypes')

    def test_counts_order_and_metadata_must_match(self):
        before, after = frames()
        with self.assertRaisesRegex(ValueError, 'cell counts'):
            vis.compute_metrics(after.iloc[:-1], MARKERS, before)
        with self.assertRaisesRegex(ValueError, 'pairing mismatch'):
            vis.compute_metrics(after.iloc[::-1].reset_index(drop=True), MARKERS, before)
        for col in ['Time', 'celltypes', 'sample_name', 'event_id']:
            bad = after.copy()
            bad[col] = bad[col].astype(object)
            bad.at[0, col] = 'wrong'
            with self.subTest(column=col), self.assertRaisesRegex(ValueError, 'pairing mismatch'):
                vis.validate_paired_data(before, bad)

    def test_weak_identity_warns_and_nondefault_index_mismatch_fails(self):
        before, after = frames()
        removed = ['event_id', 'Time', 'original_filename']
        before, after = before.drop(columns=removed), after.drop(columns=removed)
        with self.assertWarnsRegex(RuntimeWarning, 'identity cannot be verified'):
            vis.validate_paired_data(before, after)
        before.index = np.arange(1000, 1120)
        after.index = np.arange(2000, 2120)
        with self.assertRaisesRegex(ValueError, 'indices differ'):
            vis.validate_paired_data(before, after)

    def test_empty_and_missing_batch_rejected(self):
        before, after = frames()
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            vis.compute_metrics(after.iloc[:0], MARKERS, before.iloc[:0])
        with self.assertRaisesRegex(ValueError, 'column'):
            vis.validate_paired_data(before.drop(columns='plate'), after)

    def test_correction_preserves_counts_but_can_regroup_global_order(self):
        before, _ = frames()
        ctx = SimpleNamespace(meta_cols=[c for c in before if c not in MARKERS],
                              stable_marker='M3', celltypes=['T', 'B'], celltype_gating=[],
                              celltype2correct=['T', 'B'], true_markers=MARKERS, ref_plate='P1')
        with contextlib.redirect_stdout(io.StringIO()):
            intermediate, corrected = apply_correction(before.copy(), MARKERS, 'anchor', ctx)
        vis.validate_paired_data(before, corrected)
        vis.validate_paired_data(intermediate, corrected)
        shuffled = before.sample(frac=1, random_state=12).reset_index(drop=True)
        with contextlib.redirect_stdout(io.StringIO()):
            _, regrouped = apply_correction(shuffled.copy(), MARKERS, 'anchor', ctx)
        self.assertEqual(len(shuffled), len(regrouped))
        with self.assertRaisesRegex(ValueError, 'pairing mismatch'):
            vis.validate_paired_data(shuffled, regrouped)

    def test_anchor_sampler_does_not_truncate(self):
        before, after = frames()
        with self.assertRaisesRegex(ValueError, 'cell counts'):
            vis._balanced_paired_anchor_subset(before, after.iloc[:-1], 'anchor')

    def test_stratified_positions_and_categorical_missing_values(self):
        # Observe actual sampled matrices, stopping before expensive metric calculations.
        class SampleObserved(Exception):
            pass
        for categorical, missing in [(False, False), (True, False), (False, True), (True, True)]:
            before, after = frames()
            if missing:
                for frame in [before, after]:
                    frame.loc[::17, 'celltypes'] = None
                    frame.loc[::29, 'plate'] = None
            if categorical:
                for frame in [before, after]:
                    frame['plate'] = frame['plate'].astype('category')
                    frame['celltypes'] = frame['celltypes'].astype('category')
            sampled = []
            def observe(data):
                self.assertFalse(data.obs.isna().any().any())
                sampled.append(data.X[:, 0].copy())
                if len(sampled) == 2:
                    raise SampleObserved()
            with self.subTest(categorical=categorical, missing=missing):
                with mock.patch.object(vis.sc.pp, 'scale', side_effect=observe), \
                     mock.patch.object(vis.sc.tl, 'pca'), mock.patch.object(vis.sc.pp, 'neighbors'):
                    with self.assertRaises(SampleObserved):
                        vis.compute_metrics(after, MARKERS, before, subsample=40, n_iterations=1)
                np.testing.assert_array_equal(sampled[0], sampled[1])
                self.assertEqual(len(np.unique(sampled[0])), 40)
                if not missing:
                    positions = sampled[0].astype(int) - 1
                    self.assertEqual(before.iloc[positions].groupby(['plate', 'celltypes'], observed=True).size().tolist(), [10] * 4)


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name)
        patcher = mock.patch.object(vis, '_savefig', side_effect=save_small)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(vis.plt.close, 'all')

    def test_kde_normal_and_boundaries(self):
        for values in [np.zeros(20), np.ones(20), np.linspace(-2, 2, 80)]:
            distance, shift = vis.aligned_kde_js(values, values)
            self.assertAlmostEqual(distance, 0)
            self.assertAlmostEqual(shift, 0)
        for values in [[], [1], [np.nan, np.inf, -np.inf]]:
            self.assertTrue(all(np.isnan(x) for x in vis.aligned_kde_js(values, [0, 1])))
        clean = vis.aligned_kde_js([0, 1, 2], [1, 2, 3])
        dirty = vis.aligned_kde_js([0, 1, 2, np.nan, np.inf], [1, 2, 3, -np.inf])
        np.testing.assert_array_equal(clean, dirty)
        with mock.patch.object(vis, 'gaussian_filter1d', side_effect=ValueError('density failure')):
            with self.assertWarnsRegex(RuntimeWarning, 'density calculation failed'):
                self.assertTrue(np.isnan(vis.aligned_kde_js([0, 1], [1, 2])[0]))

    def test_kde_tables_plot_and_all_invalid(self):
        before, after = frames()
        table = vis.compute_kde_jsd(before, after, MARKERS, 'P1', anchor='anchor', n_sample=30, save_dir=self.out)
        self.assertEqual(len(table), len(MARKERS))
        self.assertTrue(table['KDE_JSD_raw'].between(0, 1).all())
        stats = vis.plot_kde_jsd(table, save_path=str(self.out / 'kde.png'))
        self.assertEqual(stats.iloc[0]['n_pairs'], 4)
        self.assertTrue((self.out / 'kde.png').is_file())
        pd.testing.assert_frame_equal(pd.read_csv(self.out / 'kde_jsd_results.csv'), table, check_exact=False)
        invalid = table.copy()
        invalid[['KDE_JSD_raw', 'KDE_JSD_corr']] = np.inf
        with self.assertWarnsRegex(RuntimeWarning, 'No finite'):
            stats = vis.plot_kde_jsd(invalid, save_path=str(self.out / 'invalid.png'))
        self.assertEqual(stats.iloc[0]['n_pairs'], 0)
        self.assertFalse((self.out / 'invalid.png').exists())
        self.assertTrue(vis.compute_kde_jsd(pd.DataFrame(), after, MARKERS, 'P1').empty)

    def test_emd_computation_and_existing_statistical_units(self):
        before, after = frames()
        before['cluster'] = before['celltypes']
        after['cluster'] = after['celltypes']
        table, _, mad = vis.compute_emd_mad(before, after, ['M1', 'M2', 'M3'], 'P1',
                                           cluster_col='cluster', n_sample=25, save_dir=self.out)
        self.assertEqual(len(table), 6)
        with mock.patch.object(vis, 'wilcoxon', wraps=wilcoxon) as test:
            vis.plot_emd(table, mad, save_path=self.out)
        self.assertEqual(len(test.call_args_list[0].args[0]), 6)
        self.assertEqual(len(test.call_args_list[1].args[0]), 3)
        text = (self.out / 'emd_p_value.txt').read_text()
        expected = wilcoxon(table['EMD_raw'], table['EMD_corr'], alternative='two-sided')
        self.assertIn(f"Statistic: {expected.statistic:.4f}", text)
        self.assertIn(f"P-Value: {expected.pvalue:.2e}", text)
        self.assertIn('two-sided', text)
        self.assertNotIn('Before > After', text)
        self.assertTrue((self.out / 'EMD_per_plate_boxplot.png').is_file())
        with self.assertRaisesRegex(ValueError, 'non-empty'):
            vis.plot_emd(pd.DataFrame(), mad, save_path=self.out)

    def test_emd_all_zero_differences_continue_without_tests_or_stars(self):
        mad = pd.DataFrame({'plate': ['P2'], 'MAD_score': [0.0]})
        for values in [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]:
            table = pd.DataFrame({'plate1': ['P1'] * 3, 'plate2': ['P2'] * 3,
                                  'marker': ['a', 'b', 'c'], 'SOM_node': [0] * 3,
                                  'EMD_raw': values, 'EMD_corr': values,
                                  'EMD_reduction': [0.0] * 3})
            with self.subTest(values=values), \
                 mock.patch.object(vis, 'wilcoxon') as test, \
                 mock.patch.object(vis, 'multipletests') as bh:
                vis.plot_emd(table, mad, save_path=self.out)
                test.assert_not_called()
                bh.assert_not_called()
            text = (self.out / 'emd_p_value.txt').read_text()
            self.assertIn('P-Value: nan', text)
            self.assertIn('Not testable', text)
            self.assertIn('0 valid target-plate tests', text)
            self.assertIn('excluded from BH; no stars', text)
            self.assertTrue((self.out / 'EMD_global_KDE.png').is_file())
            self.assertTrue((self.out / 'EMD_per_plate_boxplot.png').is_file())
            self.assertTrue((self.out / 'EMD_KDE_P2.png').is_file())

    def test_emd_mixed_valid_and_untestable_plates(self):
        table = pd.DataFrame({'plate1': ['P1'] * 6, 'plate2': ['P2'] * 3 + ['P3'] * 3,
                              'marker': ['a', 'b', 'c'] * 2, 'SOM_node': [0] * 6,
                              'EMD_raw': [1., 2., 3., 2., 4., 6.],
                              'EMD_corr': [1., 2., 3., 1., 2., 3.]})
        table['EMD_reduction'] = table.EMD_raw - table.EMD_corr
        mad = pd.DataFrame({'plate': ['P2', 'P3'], 'MAD_score': [0., 0.2]})
        expected = wilcoxon([2., 4., 6.], [1., 2., 3.]).pvalue
        with mock.patch.object(vis, 'wilcoxon', wraps=wilcoxon) as test, \
             mock.patch.object(vis, 'multipletests', wraps=vis.multipletests) as bh, \
             mock.patch.object(vis.plt.Axes, 'text', autospec=True) as labels:
            vis.plot_emd(table, mad, save_path=self.out)
        self.assertEqual(test.call_count, 2)  # global plus P3; P2 is untestable
        np.testing.assert_array_equal(bh.call_args.args[0], [expected])
        self.assertEqual(bh.call_args.kwargs['method'], 'fdr_bh')
        annotations = [(call.args[1], call.args[3]) for call in labels.call_args_list
                       if len(call.args) > 3 and call.args[3] in ('ns', '*', '**', '***', '****')]
        self.assertEqual(annotations, [(1, 'ns')])  # only the valid target P3
        self.assertIn('1 valid target-plate tests', (self.out / 'emd_p_value.txt').read_text())

    def test_emd_small_nonzero_differences_remain_testable(self):
        self.assertFalse(vis._emd_all_zero_finite_pairs([1., 2., 3.],
                                                       [1. + 1e-12, 2. + 1e-12, 3. + 1e-12]))

    def test_overall_plate_group_correlations_and_exports(self):
        before, after = frames()
        result = vis.marker_corr(before, after, MARKERS, group_col='celltypes',
                                  include_overall=True, save_dir=self.out)
        self.assertEqual(result['level'].value_counts().to_dict(), {'plate': 2, 'group': 2, 'overall': 1})
        self.assertTrue((self.out / 'scatter_overall.png').is_file())
        saved = pd.read_csv(self.out / 'marker_correlation_summary.csv')
        np.testing.assert_allclose(saved['R'], result['R'])
        np.testing.assert_allclose(saved['p_value_adj'], result['p_value_adj'])

    def test_boxplot_points_and_exported_values(self):
        keys = ['asw_batch', 'pcr', 'graph_connectivity', 'asw_label', 'nmi', 'ari']
        before, after = {}, {}
        for key in keys:
            values = np.array([0.3, 0.4, 0.5])
            corrected = values - 0.1 if key in ['asw_batch', 'pcr'] else values + 0.1
            before[key] = (values.mean(), values.std(), values.tolist())
            after[key] = (corrected.mean(), corrected.std(), corrected.tolist())
        vis.plot_metrics_comparison(before, after, 'Example', self.out)
        source = pd.read_csv(self.out / 'Example_metrics_subsampling_values.csv')
        self.assertEqual(len(source), 36)
        for key in keys:
            for stage, metrics in [('Before', before), ('After', after)]:
                values = source[(source.Metric_Key == key) & (source.Stage == stage)].sort_values('Iteration').Score
                np.testing.assert_allclose(values, metrics[key][2])
        summary = pd.read_csv(self.out / 'Example_metrics_summary.csv')
        np.testing.assert_allclose(summary.Before_Mean, [before[k][0] for k in keys])
        self.assertTrue((self.out / 'Example_metrics_boxplot.png').is_file())


if __name__ == '__main__':
    unittest.main()
