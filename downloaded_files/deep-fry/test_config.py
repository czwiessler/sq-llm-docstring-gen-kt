from common import TestCase

import yaml

from mayo.config import ArithTag, _DotDict, BaseConfig, Config


class TestYamlTags(TestCase):
    def test_arith_tag_construct(self):
        self.assertEqual(ArithTag('1 + 2').value(), 3)

    def test_arith_tag_from_yaml(self):
        tag = ArithTag('1 + 2')
        text = "!arith '1 + 2'"
        self.assertObjectEqual(tag, yaml.load(text))
        self.assertEqual(yaml.dump(tag).strip(), text)


class TestDotDict(TestCase):
    def setUp(self):
        self.od = {'a': 1, 'b': {'c': 2}}
        self.d = _DotDict(self.od)

    def test_simple_construct(self):
        od = {'a': 1}
        d = _DotDict(od)
        self.assertDictEqual(d._mapping, od)

    def test_incorrect_type_construct(self):
        with self.assertRaises(TypeError):
            _DotDict(1)

    def test_dot_path_construct(self):
        od = {'a.b': 1}
        d = _DotDict(od)
        self.assertDictEqual(d._mapping, {'a': {'b': 1}})

    def test_nested_dot_path_construct(self):
        od = {'a.b': {'c.d': 1}}
        d = _DotDict(od)
        td = {'a': {'b': {'c': {'d': 1}}}}
        self.assertDictEqual(d._mapping, td)
        od = {'a': {'b': {'c.d': 1}}}
        d = _DotDict(od)
        self.assertDictEqual(d._mapping, td)

    def test_merge(self):
        self.d.merge({'a': 4, 'b': {'d': 3}})
        self.assertDictEqual(self.d._mapping, {'a': 4, 'b': {'c': 2, 'd': 3}})

    def test_dot_path_merge(self):
        self.d.merge({'b.c': 3})
        self.assertDictEqual(self.d._mapping, {'a': 1, 'b': {'c': 3}})

    def test_nested_dot_path_merge(self):
        self.d.merge({'b': {'d': {'e.f': 3}}})
        self.assertDictEqual(
            self.d._mapping, {'a': 1, 'b': {'c': 2, 'd': {'e': {'f': 3}}}})

    def test_get(self):
        d = self.d
        self.assertEqual(d.a, d['a'])
        self.assertEqual(d.b.c, d['b']['c'], d['b.c'])

    def test_set(self):
        d = self.d
        d.a = 3
        self.assertEqual(d.a, d['a'], 3)
        d.b.c = 4
        self.assertEqual(d.b.c, d['b']['c'], d['b.c'], 4)

    def test_iter(self):
        self.assertListEqual(list(self.d), list(self.od))

    def test_len(self):
        self.assertCountEqual(self.d, self.od)

    def test_link(self):
        od = {'a': '$(b)'}
        link = {'b': 'c'}
        d = _DotDict(od, link)
        self.assertEqual(d['a'], link['b'])

    def test_dot_path_link(self):
        od = {'a': '$(b.c)'}
        link = _DotDict({'b': {'c': 'd'}})
        d = _DotDict(od, link)
        self.assertEqual(d['a'], link['b']['c'])

    def test_multi_hop_link(self):
        od = {'a': '$(b)', 'b': '$(c)', 'c': 'd'}
        d = _DotDict(od)
        self.assertEqual(d['a'], d['c'])

    def test_arith_tag(self):
        od = {'a': ArithTag('2')}
        d = _DotDict(od)
        self.assertEqual(d['a'], 2)


class TestBaseConfig(TestCase):
    def setUp(self):
        self.config = BaseConfig()

    def test_hook(self):
        def hook():
            raise NotImplementedError
        self.config.set('_merge_hook', {'a.b': hook})
        with self.assertRaises(NotImplementedError):
            self.config.merge({'a.b': 1})

    def test_yaml_update(self):
        self.config.yaml_update('models/lenet5.yaml')
        self.assertIn('model', self.config)

    def test_override_update(self):
        self.config.override_update('a', 1)
        self.assertEqual(self.config.a, 1)
        self.config.override_update('b.c', 2)
        self.assertEqual(self.config.b.c, 2)

    def test_yaml_export(self):
        self.assertDictEqual(
            yaml.load(self.config.to_yaml()), self.config._mapping)


class TestConfig(TestCase):
    def setUp(self):
        self.config = Config()

    def test_system(self):
        self.assertIn('system', self.config)
