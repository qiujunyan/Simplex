from fractions import Fraction
import numpy as np
from numpy import concatenate as concat
from copy import deepcopy
from functools import reduce
from pprint import pprint
import sys


class Simplex(object):
  def __init__(self, file_name):
    self.coefficients = self.get_coefficients(file_name)
    self.get_solution()

  @staticmethod
  def get_coefficients(file_path):
    fp = open(file_path)
    contents = fp.readlines()
    contents = list(map(lambda x: [float(i) for i in x.split()], contents))
    return np.array(contents)

  @staticmethod
  def print_simplex_table(coefficients, base_index):
    rows, cols = coefficients.shape
    assert len(base_index) == rows - 1
    print("{:<5}".format("BV"), end="\t")
    for col in range(1, cols):
      print("{:<5}".format(f"x_{col}"), end="\t")
    print("{:<5}".format("RHS"))

    for row in range(rows):
      if row == rows - 1:
        print("sigma", end="\t")
      else:
        print("{:<5}".format(f"x_{base_index[row] + 1}"), end="\t")
      for col in range(cols):
        print("{:<5}".format(coefficients[row, col]), end="\t")
      print()

  @staticmethod
  def find_pivot(coefficients):
    sigma = coefficients[-1, :-1]
    b = coefficients[:-1, -1]
    assert np.all(b >= 0)
    if np.all(sigma <= 0):
      print("已达到最优条件")
      return -1, -1

    # 入基变量
    col = np.where(sigma > 0)[0].min()

    # 出基变量
    row = -1
    theta = 1e32
    # p_b = deepcopy(coefficients[:-1, col])
    for i, p_bi in enumerate(coefficients[:-1, col]):
      if p_bi <= 0:
        continue
      tmp = b[i] / p_bi
      if tmp < theta:
        row = i
        theta = tmp
    if row < 0:
      print("无界解")
      sys.exit()
    return row, col

  @staticmethod
  def row_operation(coefficients, row, col):
    m, n = coefficients.shape
    for i in range(m):
      if i == row:
        coefficients[i] = coefficients[i] / (coefficients[row, col])
      else:
        coefficients[i] = coefficients[i] - \
                          (coefficients[i, col] / coefficients[row, col]) * coefficients[row]
    # return coefficients

  def get_init_feasible_solution(self):
    print("1.两阶段法求初始可行解")
    m, n = self.coefficients.shape
    coefficients = np.zeros([m, m + n - 1])
    coefficients[:-1, :n - 1] = self.coefficients[:-1, :n - 1]  # 原问题系数矩阵
    coefficients[:-1, -1] = self.coefficients[:-1, -1]  # 原问题b
    coefficients[:-1, n - 1:-1] = np.eye(m - 1)  # 人工变量系数矩阵
    coefficients[-1, n - 1:-1] = -1  # 人工变量检验数
    # 基变量对应检验数变成0
    for i in range(m - 1):
      coefficients[-1] += coefficients[i]

    base_index = list(range(n - 1, m + n - 2))
    self.print_simplex_table(coefficients, base_index)

    while np.any(coefficients[-1, :-1] > 0):
      row, col = self.find_pivot(coefficients)
      self.row_operation(coefficients, row, col)
      print(f"x_{base_index[row] + 1}出基，x_{col + 1}进基")
      base_index[row] = col
      self.print_simplex_table(coefficients, base_index)

    if coefficients[-1, -1] == 0:
      print("找到初始可行解")
      self.coefficients[:-1, :n - 1] = coefficients[:-1, :n - 1]
      self.coefficients[:-1, -1] = coefficients[:-1, -1]

      for row, col in enumerate(base_index):
        self.row_operation(self.coefficients, row, col)
      return self.coefficients, base_index

    else:
      print("无初始可行解")

  def get_solution(self):
    self.coefficients, base_index = self.get_init_feasible_solution()
    print("*" * 100 + "\n" + "init:")
    self.print_simplex_table(self.coefficients, base_index)
    while np.any(self.coefficients[-1, :-1] > 0):
      row, col = self.find_pivot(self.coefficients)
      self.row_operation(self.coefficients, row, col)
      print(f"x_{base_index[row] + 1}出基，x_{col + 1}进基")
      base_index[row] = col
      self.print_simplex_table(self.coefficients, base_index)
    num_vars = self.coefficients.shape[1] - 1
    solution = [0] * num_vars
    for i, index in enumerate(base_index):
      solution[index] = self.coefficients[i, -1]

    print("最优解为：{}".format(str(tuple(solution))))


if __name__ == '__main__':
  simplex = Simplex("coefficients.txt")
  # simplex.get_init_feasible_solution()
