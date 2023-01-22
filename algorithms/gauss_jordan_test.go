package algorithms

import (
	"testing"

    "gonum.org/v1/gonum/mat"
    "github.com/james-bowman/sparse"
)

var TwoVariableSystem *mat.Dense = mat.NewDense(2, 3, []float64{
    2, 2, 4,
    4, 8, 14,
})

var TwoVariableSolution *mat.Dense = mat.NewDense(2, 3, []float64{
    1, 0, 0.5,
    0, 1, 1.5,
})

func TestTwoVariableSystem(t *testing.T) {
    solution := GaussJordan(TwoVariableSystem)
    fa := mat.Formatted(solution.Matrix, mat.Prefix("    "), mat.Squeeze())
    if solutionType := solution.Type(); !mat.Equal(solution.Matrix, TwoVariableSolution) || solutionType != UniqueGaussJordanSolution{
        t.Errorf("Computed incorrect solution (%d):\na = % v\n\n", solutionType, fa)
    }
}

var TwoVariableSystemForBench *mat.Dense = mat.NewDense(2, 3, []float64{
    2, 2, 4,
    4, 8, 14,
})

func BenchmarkTwoVariableSystem(b *testing.B) {
    for n := 0; n < b.N; n++ {
        GaussJordan(TwoVariableSystemForBench)
    }
}


var TwoVariableNoSolutionSystem *mat.Dense = mat.NewDense(2, 3, []float64{
    1, 2, 3,
    2, 4, 4,
})

var TwoVariableNoSolution *mat.Dense = mat.NewDense(2, 3, []float64{
    1, 2, 3,
    0, 0, -2,
})


func TestTwoVariableNoSolutionSystem(t *testing.T) {
    solution := GaussJordan(TwoVariableNoSolutionSystem)
    fa := mat.Formatted(solution.Matrix, mat.Prefix("    "), mat.Squeeze())
    if !mat.Equal(solution.Matrix, TwoVariableNoSolution) || solution.Type() != NoGaussJordanSolutions {
        t.Errorf("Computed incorrect solution:\na = % v\n\n", fa)
    }
}



func denseToCSR(matrix *mat.Dense) *CSRMatrix  {
    m, n := matrix.Dims()
    result := sparse.NewDOK(m, n)
    for i := 0; i < m; i++ {
        for j := 0; j < n; j++ {
            result.Set(i, j, matrix.At(i, j))
        }
    }
    return &CSRMatrix{result.ToCSR()}
}



var SparseTwoVariableSystem *CSRMatrix = denseToCSR(TwoVariableSystem)

var SparseTwoVariableSolution *CSRMatrix = denseToCSR(TwoVariableSolution)

func TestSparseTwoVariableSystem(t *testing.T) {
    solution := GaussJordan(SparseTwoVariableSystem)
    fa := mat.Formatted(solution.Matrix, mat.Prefix("    "), mat.Squeeze())
    if !mat.Equal(solution.Matrix, SparseTwoVariableSolution){
        t.Errorf("Computed incorrect solution:\na = % v\n\n", fa)
    }
}

var SparseTwoVariableSystemForBench *CSRMatrix = denseToCSR(TwoVariableSystem)

func BenchmarkSparseTwoVariableSystem(b *testing.B) {
    for n := 0; n < b.N; n++ {
        GaussJordan(SparseTwoVariableSystemForBench)
    }
}