from imports import *

class AHPBackend:
    # Константы для проверки значений по шкале Саати
    VALID_SAATY_VALUES = {1, 2, 3, 4, 5, 6, 7, 8, 9}
    RI_VALUES = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12,
                 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45,
                 10: 1.49, 11: 1.51, 12: 1.54, 13: 1.56,
                 14: 1.57, 15: 1.59}


    def __init__(self):
        self.reset_all_data()


    def reset_all_data(self):
        # Сброс всех данных для нового расчета
        self.alternatives: List[str] = []
        self.criteria: List[str] = []
        self.criteria_types: Dict[str, List[str]] = {}
        self.matrices: Dict[str, np.ndarray] = {}
        self.priorities: Dict[str, np.ndarray] = {}
        self.consistency_data: Dict[str, Dict[str, float]] = {}


    def add_alternative(self, name: str) -> bool:
        # Добавляет альтернативу
        name = name.strip()
        if name and name not in self.alternatives:
            self.alternatives.append(name)
            return True
        return False


    def add_criterion(self, name: str) -> bool:
        # Добавляет критерий
        name = name.strip()
        if name and name not in self.criteria:
            self.criteria.append(name)
            return True
        return False


    def add_criterion_type(self, type_name: str, criteria: List[str]) -> bool:
        """Добавляет тип критериев"""
        type_name = type_name.strip()
        if not type_name or not criteria:
            return False

        valid_criteria = [c for c in criteria if c in self.criteria]
        if valid_criteria:
            self.criteria_types[type_name] = valid_criteria
            return True
        return False


    def validate_matrix_value(self, value: str) -> bool:
        """Проверяет значение по шкале Саати"""
        try:
            if value.startswith("1/"):
                num = float(value[2:])
                return num in self.VALID_SAATY_VALUES
            num = float(value)
            return num in self.VALID_SAATY_VALUES
        except (ValueError, AttributeError):
            return False

    def build_matrix(self, items: List[str], comparisons: Dict[Tuple[int, int], str]) -> Optional[np.ndarray]:
        # Строит матрицу парных сравнений
        n = len(items)
        if n == 0:
            return None

        matrix = np.eye(n)
        for (i, j), value in comparisons.items():
            if not (0 <= i < n and 0 <= j < n) or not self.validate_matrix_value(value):
                return None

            try:
                if value.startswith("1/"):
                    val = 1 / float(value[2:])
                else:
                    val = float(value)

                matrix[i, j] = val
                matrix[j, i] = 1 / val
            except (ValueError, ZeroDivisionError):
                return None

        return matrix


    def calculate_priority_vector(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Вычисляет главный вектор и вектор приоритетов
        n = matrix.shape[0]
        CB = np.array([np.prod(matrix[i, :]) ** (1 / n) for i in range(n)])
        w = CB / np.sum(CB)
        return CB, w

    def calculate_ahp(self, selected_levels: int = 3) -> Dict[
        str, Union[Dict[str, np.ndarray], Dict[str, Dict[str, float]], List[str]]]:
        # Основной метод расчета МАИ с учетом уровней иерархии
        results = {
            'priorities': {},
            'consistency': {},
            'errors': [],
            'matrix_count': 0
        }

        try:
            # 1. Расчет для типов критериев (только для 3 уровней)
            if selected_levels >= 3:
                if not self.criteria_types:
                    results['errors'].append("Не заданы типы критериев для 3-уровневой иерархии")
                    return results

                type_matrix = self.matrices.get('criteria_types')
                if type_matrix is None:
                    results['errors'].append("Отсутствует матрица сравнения типов критериев")
                    return results

                results['matrix_count'] += 1
                CB_types, w_types = self.calculate_priority_vector(type_matrix)
                results['priorities']['type_priority'] = w_types
                results['consistency']['criteria_types'] = self.check_consistency(type_matrix)

            # 2. Расчет для критериев (для 2 и 3 уровней)
            if selected_levels >= 2:
                if not self.criteria:
                    results['errors'].append("Не заданы критерии для расчета")
                    return results

                criteria_priority = np.zeros(len(self.criteria))

                if selected_levels >= 3:
                    # Для 3 уровней - расчет по типам критериев
                    type_names = list(self.criteria_types.keys())
                    for type_name, type_criteria in self.criteria_types.items():
                        criteria_matrix = self.matrices.get(f'criteria_{type_name}')
                        if criteria_matrix is None:
                            results['errors'].append(f"Отсутствует матрица сравнения критериев для типа '{type_name}'")
                            continue

                        results['matrix_count'] += 1
                        CB_criteria, w_criteria = self.calculate_priority_vector(criteria_matrix)

                        # Нормализуем веса критериев внутри типа
                        w_criteria_normalized = w_criteria / np.sum(w_criteria)

                        # Умножаем на вес типа (матричное умножение)
                        criteria_priority[[self.criteria.index(c) for c in type_criteria]] = (
                                w_criteria_normalized * w_types[type_names.index(type_name)]
                        )

                        results['consistency'][f'criteria_{type_name}'] = self.check_consistency(criteria_matrix)

                    # Нормализуем итоговые веса критериев
                    sum_criteria = np.sum(criteria_priority)
                    if sum_criteria == 0:
                        results['errors'].append("Суммарный вес критериев равен нулю")
                        return results
                    criteria_priority = criteria_priority / sum_criteria
                else:
                    # Для 2 уровней - простой расчет (как для первого уровня)
                    criteria_matrix = self.matrices.get('criteria')
                    if criteria_matrix is None:
                        results['errors'].append("Отсутствует матрица сравнения критериев")
                        return results

                    results['matrix_count'] += 1
                    CB_criteria, criteria_priority = self.calculate_priority_vector(criteria_matrix)
                    results['consistency']['criteria'] = self.check_consistency(criteria_matrix)

                results['priorities']['criteria_priority'] = criteria_priority

            # 3. Расчет для альтернатив (для всех уровней)
            if not self.alternatives:
                results['errors'].append("Не заданы альтернативы для расчета")
                return results

            alternatives_priority = np.zeros(len(self.alternatives))

            if selected_levels >= 2:
                # Для 2 и 3 уровней - расчет по критериям
                for criterion in self.criteria:
                    alt_matrix = self.matrices.get(f'alternatives_{criterion}')
                    if alt_matrix is None:
                        results['errors'].append(
                            f"Отсутствует матрица сравнения альтернатив для критерия '{criterion}'")
                        continue

                    results['matrix_count'] += 1
                    CB_alt, w_alt = self.calculate_priority_vector(alt_matrix)

                    # Нормализуем веса альтернатив для каждого критерия
                    w_alt_normalized = w_alt / np.sum(w_alt) if np.sum(w_alt) != 0 else w_alt

                    # Умножаем на вес критерия (матричное умножение)
                    alternatives_priority += w_alt_normalized * criteria_priority[self.criteria.index(criterion)]

                    results['consistency'][f'alternatives_{criterion}'] = self.check_consistency(alt_matrix)
            else:
                # Для 1 уровня - простой расчет (как для первого уровня)
                alt_matrix = self.matrices.get('alternatives')
                if alt_matrix is None:
                    results['errors'].append("Отсутствует матрица сравнения альтернатив")
                    return results

                results['matrix_count'] += 1
                CB_alt, alternatives_priority = self.calculate_priority_vector(alt_matrix)
                results['consistency']['alternatives'] = self.check_consistency(alt_matrix)

            # Финальная нормализация весов альтернатив
            sum_alternatives = np.sum(alternatives_priority)
            if sum_alternatives == 0:
                results['errors'].append("Суммарный вес альтернатив равен нулю")
                return results
            alternatives_priority = alternatives_priority / sum_alternatives

            results['priorities']['alternatives_priority'] = alternatives_priority
            return results

        except Exception as e:
            results['errors'].append(f"Ошибка расчета: {str(e)}")
            return results


    def check_consistency(self, matrix: np.ndarray) -> Dict[str, float]:
        # Проверка согласованности матрицы
        n = matrix.shape[0]
        if n <= 2:
            return {
                'lambda_max': float(n),
                'CI': 0.0,
                'RI': 0.0,
                'CR': 0.0,
                'status': 'Отличная согласованность'
            }

        CB, w = self.calculate_priority_vector(matrix)
        weighted_sum = np.dot(matrix, w)
        lambda_max = np.mean(weighted_sum / w)
        CI = (lambda_max - n) / (n - 1)
        RI = self.RI_VALUES.get(n, 1.49)
        CR = CI / RI if RI != 0 else 0

        status = "Отличная согласованность" if CR < 0.1 else \
            "Приемлемая согласованность" if CR < 0.2 else \
                "ТРЕБУЕТСЯ пересмотр"

        return {
            'lambda_max': round(lambda_max, 3),
            'CI': round(CI, 3),
            'RI': round(RI, 3),
            'CR': round(CR, 3),
            'status': status
        }
