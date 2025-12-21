import numpy as np
import pytest
from sklearn.utils.validation import NotFittedError

from skrub import _dataframe as sbd
from skrub._single_column_transformer import (
    RejectColumn,
    SingleColumnTransformer,
    _insert_after_first_paragraph,
    _wrap_add_check_single_column,
)


class TestRejectColumn:
    """Test the RejectColumn exception."""

    def test_reject_column_is_value_error(self):
        """RejectColumn should be a subclass of ValueError."""
        assert issubclass(RejectColumn, ValueError)

    def test_reject_column_can_be_raised_and_caught(self):
        """Test that RejectColumn can be raised and caught as expected."""
        with pytest.raises(RejectColumn, match="test message"):
            raise RejectColumn("test message")

    def test_reject_column_without_message(self):
        """Test that RejectColumn can be raised without a message."""
        with pytest.raises(RejectColumn):
            raise RejectColumn()


class TestSingleColumnTransformer:
    """Test the SingleColumnTransformer base class."""

    def test_has_single_column_transformer_attribute(self):
        """The class should have __single_column_transformer__ = True."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

        assert DummyTransformer.__single_column_transformer__ is True
        assert DummyTransformer().__single_column_transformer__ is True

    def test_fit_calls_fit_transform(self, df_module):
        """The default fit() should call fit_transform()."""

        class DummyTransformer(SingleColumnTransformer):
            def __init__(self):
                self.fit_transform_called = False

            def fit_transform(self, column, y=None, **kwargs):
                self.fit_transform_called = True
                self.fitted_ = True
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()
        col = df_module.example_column
        result = transformer.fit(col)

        assert result is transformer
        assert transformer.fit_transform_called
        assert transformer.fitted_

    def test_check_single_column_with_dataframe(self, df_module):
        """_check_single_column should raise error when passed a dataframe."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()
        df = df_module.example_dataframe

        with pytest.raises(
            ValueError,
            match=r"``DummyTransformer\.fit`` should be passed a single column",
        ):
            transformer.fit(df)

    def test_check_single_column_with_invalid_type(self, df_module):
        """_check_single_column should raise error for non-column types."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()

        with pytest.raises(
            ValueError,
            match=r"``DummyTransformer\.fit`` expects the first argument X",
        ):
            transformer.fit(np.array([1, 2, 3]))

        with pytest.raises(
            ValueError,
            match=r"``DummyTransformer\.transform`` expects the first argument X",
        ):
            transformer.transform([1, 2, 3])

    def test_fit_transform_is_wrapped(self, df_module):
        """fit_transform should be wrapped with _check_single_column."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()
        df = df_module.example_dataframe

        with pytest.raises(
            ValueError,
            match=(
                r"``DummyTransformer\.fit_transform`` "
                r"should be passed a single column"
            ),
        ):
            transformer.fit_transform(df)

    def test_transform_is_wrapped(self, df_module):
        """transform should be wrapped with _check_single_column."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                self.fitted_ = True
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()
        col = df_module.example_column
        transformer.fit(col)

        df = df_module.example_dataframe
        with pytest.raises(
            ValueError,
            match=r"``DummyTransformer\.transform`` should be passed a single column",
        ):
            transformer.transform(df)

    def test_partial_fit_is_wrapped(self, df_module):
        """partial_fit should be wrapped with _check_single_column if defined."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

            def partial_fit(self, column, y=None):
                return self

        transformer = DummyTransformer()
        df = df_module.example_dataframe

        with pytest.raises(
            ValueError,
            match=r"``DummyTransformer\.partial_fit`` should be passed a single column",
        ):
            transformer.partial_fit(df)

    def test_custom_fit_is_wrapped(self, df_module):
        """If subclass defines custom fit(), it should also be wrapped."""

        class DummyTransformer(SingleColumnTransformer):
            def fit(self, column, y=None, **kwargs):
                self.fitted_ = True
                return self

            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()
        df = df_module.example_dataframe

        with pytest.raises(
            ValueError,
            match=r"``DummyTransformer\.fit`` should be passed a single column",
        ):
            transformer.fit(df)

    def test_get_feature_names_out_not_fitted(self):
        """get_feature_names_out should raise error when not fitted."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()

        with pytest.raises(NotFittedError):
            transformer.get_feature_names_out()

    def test_get_feature_names_out(self, df_module):
        """get_feature_names_out should return all_outputs_ when fitted."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                self.all_outputs_ = ["output1", "output2"]
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()
        col = df_module.example_column
        transformer.fit(col)

        assert transformer.get_feature_names_out() == ["output1", "output2"]

    def test_docstring_modification(self):
        """Test that docstrings are modified to include single-column note."""

        class WithDocstring(SingleColumnTransformer):
            """A simple transformer.

            This transformer does something.
            """

            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

        assert "A simple transformer." in WithDocstring.__doc__
        assert (
            "``WithDocstring`` is a type of single-column transformer"
            in WithDocstring.__doc__
        )
        assert "This transformer does something." in WithDocstring.__doc__

    def test_docstring_none(self):
        """Test that None docstrings are handled correctly."""

        class NoDocstring(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                return column

            def transform(self, column):
                return column

        assert NoDocstring.__doc__ is None

    def test_works_with_y_parameter(self, df_module):
        """Test that y parameter is passed correctly through methods."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                self.y_ = y
                return column

            def transform(self, column):
                return column

        transformer = DummyTransformer()
        col = df_module.example_column
        y = df_module.make_column("target", [1, 2, 3])

        transformer.fit(col, y=y)
        assert transformer.y_ is y

    def test_works_with_kwargs(self, df_module):
        """Test that extra kwargs are passed through correctly."""

        class DummyTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None, **kwargs):
                self.kwargs_ = kwargs
                return column

            def transform(self, column, **kwargs):
                return column

        transformer = DummyTransformer()
        col = df_module.example_column

        transformer.fit(col, extra_param="value")
        assert transformer.kwargs_ == {"extra_param": "value"}


class TestInsertAfterFirstParagraph:
    """Test the _insert_after_first_paragraph helper function."""

    def test_simple_insertion(self):
        """Test insertion into a simple docstring."""
        doc = "Summary line.\n\nDetails here."
        result = _insert_after_first_paragraph(doc, "Inserted text.\n")
        expected = "Summary line.\n\nInserted text.\n\nDetails here."
        assert result == expected

    def test_insertion_with_indentation(self):
        """Test that indentation is preserved."""
        doc = "    Summary line.\n\n    Details here."
        result = _insert_after_first_paragraph(doc, "Inserted.\n")
        assert "    Inserted.\n" in result

    def test_single_line_docstring(self):
        """Test insertion for single-line docstring."""
        doc = "Summary only."
        result = _insert_after_first_paragraph(doc, "Inserted.\n")
        assert "Inserted.\n" in result

    def test_multiline_summary(self):
        """Test insertion after multi-line summary."""
        doc = "Summary line one.\n    Summary line two.\n\n    Details."
        result = _insert_after_first_paragraph(doc, "Inserted.\n")
        assert "Summary line two.\n\n    Inserted.\n\n    Details." in result

    def test_empty_lines_handling(self):
        """Test that empty lines are handled correctly."""
        doc = "\n    Summary.\n\n    Details.\n"
        result = _insert_after_first_paragraph(doc, "Inserted.\n")
        assert "    Inserted.\n" in result


class TestWrapAddCheckSingleColumn:
    """Test the _wrap_add_check_single_column helper function."""

    def test_wrap_fit(self, df_module):
        """Test wrapping of fit method."""

        def fit(self, X, y=None, **kwargs):
            self.fitted_ = True
            return self

        class Dummy:
            _check_single_column = lambda self, col, name: col

        wrapped_fit = _wrap_add_check_single_column(fit)
        dummy = Dummy()
        col = df_module.example_column

        result = wrapped_fit(dummy, col)
        assert result is dummy
        assert dummy.fitted_

    def test_wrap_transform(self, df_module):
        """Test wrapping of transform method."""

        def transform(self, X, **kwargs):
            return X

        class Dummy:
            _check_single_column = lambda self, col, name: col

        wrapped_transform = _wrap_add_check_single_column(transform)
        dummy = Dummy()
        col = df_module.example_column

        result = wrapped_transform(dummy, col)
        assert result is col

    def test_wrap_fit_transform(self, df_module):
        """Test wrapping of fit_transform method."""

        def fit_transform(self, X, y=None, **kwargs):
            self.fitted_ = True
            return X

        class Dummy:
            _check_single_column = lambda self, col, name: col

        wrapped_fit_transform = _wrap_add_check_single_column(fit_transform)
        dummy = Dummy()
        col = df_module.example_column

        result = wrapped_fit_transform(dummy, col, y=None)
        assert result is col
        assert dummy.fitted_

    def test_wrap_partial_fit(self, df_module):
        """Test wrapping of partial_fit method."""

        def partial_fit(self, X, y=None, **kwargs):
            self.partial_fitted_ = True
            return self

        class Dummy:
            _check_single_column = lambda self, col, name: col

        wrapped_partial_fit = _wrap_add_check_single_column(partial_fit)
        dummy = Dummy()
        col = df_module.example_column

        result = wrapped_partial_fit(dummy, col)
        assert result is dummy
        assert dummy.partial_fitted_

    def test_wrapped_function_preserves_name(self):
        """Test that wrapped functions preserve their name."""

        def fit(self, X, y=None):
            return self

        wrapped = _wrap_add_check_single_column(fit)
        assert wrapped.__name__ == "fit"

        def transform(self, X):
            return X

        wrapped = _wrap_add_check_single_column(transform)
        assert wrapped.__name__ == "transform"


class TestRejectColumnIntegration:
    """Test RejectColumn in realistic scenarios."""

    def test_reject_column_can_be_caught_selectively(self, df_module):
        """Test that RejectColumn can be caught while other errors propagate."""

        class SelectiveTransformer(SingleColumnTransformer):
            def fit_transform(self, column, y=None):
                col_name = sbd.name(column)
                if "reject" in col_name:
                    raise RejectColumn(f"Column {col_name} is rejected")
                if "error" in col_name:
                    raise RuntimeError(f"Error in column {col_name}")
                return column * 2

            def transform(self, column):
                return column * 2

        transformer = SelectiveTransformer()

        # Should raise RejectColumn
        reject_col = df_module.make_column("reject_me", [1, 2, 3])
        with pytest.raises(RejectColumn, match="Column reject_me is rejected"):
            transformer.fit_transform(reject_col)

        # Should raise RuntimeError
        error_col = df_module.make_column("error_me", [1, 2, 3])
        with pytest.raises(RuntimeError, match="Error in column error_me"):
            transformer.fit_transform(error_col)

        # Should work normally
        normal_col = df_module.make_column("normal", [1.0, 2.0, 3.0])
        result = transformer.fit_transform(normal_col)
        expected = df_module.make_column("normal", [2.0, 4.0, 6.0])
        df_module.assert_column_equal(result, expected)
