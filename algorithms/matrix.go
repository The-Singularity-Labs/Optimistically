package algorithms

import (
	"gonum.org/v1/gonum/mat"
	"github.com/james-bowman/sparse"
)

type Matrix interface {
	mat.Matrix
	Set(m, n int, v float64)
	Mul(a, b mat.Matrix)
	SetRow(i int, src []float64)
	RowView(i int) mat.Vector
}

type CSRMatrix struct {
    *sparse.CSR
}

func (matrix *CSRMatrix) Mul(a, b mat.Matrix) {
	am, _ := a.Dims()
	_, bn := b.Dims()
	var result *sparse.CSR = sparse.NewCSR(am, bn, []int{}, []int{}, []float64{})
	result.Mul(a, b)
	matrix.CSR = result
}

func (matrix *CSRMatrix) SetRow(i int, src []float64) {
	for j, val := range src {
		matrix.CSR.Set(i, j, val)
	}
}

