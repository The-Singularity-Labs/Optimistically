package algorithms

import (
	"gonum.org/v1/gonum/mat"
)

type GaussJordanSolutionType int64

const (
     NoGaussJordanSolutions GaussJordanSolutionType = 0
     UniqueGaussJordanSolution = 1
     InfiniteGaussJordanSolutions = 2
 )

type GaussJordanSolution struct {
	Matrix mat.Matrix `json:"matrix"`
}

func (solution GaussJordanSolution) IsLinearlyIndependent() bool {
	m, _ := solution.Matrix.Dims()
	return solution.Rank() == m
}

func (solution GaussJordanSolution) Rank() int { 
	m, n := solution.Matrix.Dims()
	foundZeroRows := 0
	for i := 0; i < m; i++ {
		currentRow := make([]float64, n)
		mat.Row(currentRow, i, solution.Matrix)
		if allZeros(currentRow, len(currentRow) -1) && currentRow[len(currentRow) -1] != 0 {
			foundZeroRows = foundZeroRows + 1
			break
		}
	}
	return m - foundZeroRows
}

func (solution GaussJordanSolution) Type() GaussJordanSolutionType {
	m, n := solution.Matrix.Dims()
	foundZeroRow := false
	foundNonIdentityRow := false
	for i := 0; i < m; i++ {
		currentRow := make([]float64, n)
		mat.Row(currentRow, i, solution.Matrix)
		if allZeros(currentRow, len(currentRow) -1) && currentRow[len(currentRow) -1] != 0 {
			foundZeroRow = true
			break
		}
		if i >= len(currentRow) || currentRow[i] != 1 || !allZeros(currentRow[:len(currentRow) -1], i){
			foundNonIdentityRow = true
		}
	}
	if foundZeroRow {
		return NoGaussJordanSolutions
	} else if foundNonIdentityRow {
		return InfiniteGaussJordanSolutions
	} else {
		return UniqueGaussJordanSolution
	}
}

func scalarMultiply(vec []float64, scalar float64) []float64 {
	for i := 0; i < len(vec); i++ {
		vec[i] = vec[i] * scalar
	}
	return vec
}

func vectorAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("Could not handle vectors of differing lengths")
	}
	for i := 0; i < len(a); i++ {
		a[i] = a[i] + b[i]
	}
	return a
}

func eroTransform(augmentedMatrix Matrix, i int) Matrix {
	m, n := augmentedMatrix.Dims()
	currentEntry := augmentedMatrix.At(i, i)
	if currentEntry == 0 {
		foundSwap := false
		for j := i + 1; j < m; j++ {
			if otherEntry := augmentedMatrix.At(j, i); otherEntry != 0 {
				currentRow := make([]float64, n)
				betterRow := make([]float64, n)
				mat.Row(currentRow, i, augmentedMatrix)
				mat.Row(betterRow, j, augmentedMatrix)
				augmentedMatrix.SetRow(j, currentRow)
				augmentedMatrix.SetRow(i, betterRow)
			}
		}
		if !foundSwap{
			return augmentedMatrix
		}
	}
	if currentEntry != 1 {
		scale := mat.NewDense(m, m, nil)

		for j := 0; j < m; j++ {
			if j == i {
				scale.Set(i, i, 1 / currentEntry)
			} else {
				scale.Set(j, j, 1.0)
			}
		}
		augmentedMatrix.Mul(scale, augmentedMatrix)
	}
	for j := 0; j < m; j++ {
		if j != i {
			if otherEntry := augmentedMatrix.At(j, i); otherEntry != 0 {
				currentRow := make([]float64, n)
				existingRow := make([]float64, n)
				mat.Row(currentRow, i, augmentedMatrix)
				mat.Row(existingRow, j, augmentedMatrix)
				augmentedMatrix.SetRow(j, vectorAdd(existingRow, scalarMultiply(currentRow, (-1 * otherEntry))))
			}
		}
	}
	return augmentedMatrix
}

func allZeros(lst []float64, ignore int) (isAllZeros bool){
	isAllZeros = true
	for idx, elem := range lst {
		if idx != ignore && elem != 0 {
			isAllZeros = false
			break
		}
	}
	return
}


func GaussJordan(augmentedMatrix Matrix) GaussJordanSolution {
	m, n := augmentedMatrix.Dims()
	for i := 0; i < n - 1; i++ {
		if i < m {
			augmentedMatrix = eroTransform(augmentedMatrix, i)
		}
	}
	return GaussJordanSolution{Matrix: augmentedMatrix}
}
