import numpy as np
import streamlit as st, time
from sympy import symbols, sympify, lambdify
import pandas as pd
from plotly import express as px
from Calculations.ErrorCalculator import ErrorCalculator
from Calculations.NormCalculator import NormCalculator
from Calculations.OtherCalculator import Calculations
from FiniteElementsMethod.BoundaryCondition import DirichletCondition, RobinCondition
from FiniteElementsMethod.BoundaryProblem import BoundaryProblem
from FiniteElementsMethod.Discretizer import Discretizer
from FiniteElementsMethod.ParabolicBoundaryProblem import ParabolicBoundaryProblem
from FiniteElementsMethod.ParabolicDiscretizer import ParabolicDiscretizer
from FiniteElementsMethod.TridiagonalMatrixAlgorithm import TridiagonalMatrixAlgorithm



class Data:
    def coefficients_inputs(self):
        self._ro = st.text_input("ro:")
        self._mu = st.text_input("mu:")
        self._beta = st.text_input("beta:")
        self._sigma = st.text_input("sigma:")
        self._f = st.text_input("f:")
        self._c = st.text_input("c:")
        self._teta = st.text_input("teta:")

    @staticmethod
    def boundary_selectors(lcol, rcol):
        selected = {}
        bc_options = ['Dirichlet', 'Neumann', 'Robin']
        with lcol:
            selected["left"] = st.selectbox("Left boundary condition", bc_options)
        with rcol:
            selected["right"] = st.selectbox("Right boundary condition", bc_options)
        return selected

    def boundary_input_form(self, selected, lcol, rcol):
        with lcol:
            self._leftBC = Data.display_bc_inputs(selected["left"], "left")
        with rcol:
            self._rightBC = Data.display_bc_inputs(selected["right"], "right")


    def create_boundary_problem(self):
        x, t = symbols('x t')
        mu = lambdify([x, t], sympify(self._mu))
        beta = lambdify([x, t], sympify(self._beta))
        sigma = lambdify([x, t], sympify(self._sigma))
        f = lambdify([x, t], sympify(self._f))
        ro = lambdify([x, t], sympify(self._ro))
        c = lambdify(x, sympify(self._c))

        leftBc = Data.build_pdecondition(self._leftBC, t)
        rightBC = Data.build_pdecondition(self._rightBC, t)

        st.session_state['BoundaryProblem'] = ParabolicBoundaryProblem(ro, mu, beta, sigma, f, c, leftBc, rightBC)
        st.success('Boundary problem was successfully defined')


    @staticmethod
    def build_pdecondition(bc, x):
        bcType = bc['type']
        if bcType == 'Dirichlet':
            return DirichletCondition(lambdify(x, sympify(bc['u'])))
        elif bcType in ['Neumann', 'Robin']:
            return RobinCondition(lambdify(x, sympify(bc['g'])), lambdify(x, sympify(bc['q'])))

    @staticmethod
    def display_bc_inputs(selected, side):
        values = {}
        if selected == "Dirichlet":
            values['type'] = "Dirichlet"
            values['u'] = st.text_input("u()=", key=f'uvalue_{side}_input')
        elif selected == "Neumann":
            values['type'] = "Neumann"
            values['g'] = st.text_input("g=" , key=f'gvalue_{side}_input')
            values['q'] = '0'
        elif selected == "Robin":
            values['type'] = "Robin"
            values['g'] = st.text_input("g=", key=f'gvalue_{side}_input')
            values['q'] = st.text_input("q=", key=f'qvalue_{side}_input')
        return values

    def FEM(self):
        with st.form(key='fem_inputs'):
            self._n = st.slider('Initial number of nodes:', key='nodes_count_input',
                                       min_value=3, max_value=15, value=3, step=1)
            self._t = st.slider('Time', key='time_count_input',
                                                           min_value=1, max_value=30, value=1, step=1)
            self._u = st.text_input('True solution (Optional):', help='Optional', key='truesolution_input')
            is_submitted = st.form_submit_button('Ok', type='primary')
            if is_submitted:
                self.build_dicretizer()

    def build_dicretizer(self):
        pdeproblem = st.session_state['BoundaryProblem']
        st.session_state['discretizer'] = ParabolicDiscretizer(pdeproblem)
        st.session_state['nodescount'] = self._n
        st.session_state['timecount'] = self._t
        if self._u:
            x = symbols('x')
            st.session_state['true_solution'] = lambdify(x, sympify(self._u))
        elif 'true_solution' in st.session_state:
            del st.session_state['true_solution']


    def iterations(self):

        femdiscretizer = st.session_state['discretizer']
        calculator = Calculations(st.session_state['BoundaryProblem'])
        errors = ErrorCalculator(st.session_state['BoundaryProblem'])
        norms = NormCalculator(st.session_state['BoundaryProblem'])
        
        with st.spinner('Computing...'):
            time.sleep(0.5)
            # nodes_now = self._n
            # nodes, K, l = femdiscretizer.do_discretization(nodes_now)
            # alpha = TridiagonalMatrixAlgorithm(K, l).solve()
            #
            #
            # iterations = []
            # nodes_count_now = []
            # r_value = []
            # j_value = []
            # neuman_errors = []
            # dirich_errors = []
            # energy = []
            # sobol = []
            # q_value = []
            # u_sobol = []
            # u_energy =[]


            energy = []
            sobol = []
            neuman_errors = []
            t = []
            nodes_count_now = []
            dirich_errors = []
            # t.append(0)
            nodes = femdiscretizer.buildMesh(self._n)
            nod, _ = nodes
            q0 = femdiscretizer.buildInitialSolution(nodes)
            qs = [q0]
            dT = 2.0/(st.session_state['timecount'] - 1)
            #  дельта т
            for i in range(1, self._t):
                ti = (i-0.5) * dT
                M, A, l = femdiscretizer.discretize(nodes, ti)
                # femdiscretizer.check(A, M, l, dT, float(self._teta), ti)
                dqi = TridiagonalMatrixAlgorithm(M+dT * float(self._teta) * A, l-femdiscretizer.mmul(A, qs[-1])).solve()
                qi = qs[-1] + dqi * dT
                qs.append(qi)


                t.append(ti)
                nodes_count_now.append(self._n)
                energy.append(norms.energy_norm(nod, qi))
                neuman_errors.append(errors.neumann_error(nod, qi, ti))
                dirich_errors.append(errors.dirichlet_error(nod, qi, ti))
                sobol.append(norms.sobolev_norm(nod, qi))






            df = pd.DataFrame(qs).T
            df['x'], _ = nodes
            df = df.melt(id_vars='x', var_name='t', value_name='uh')
            df['t'] *= np.round(dT, 2)
            # t = df['t'] * np.round(dT, 2)

            data2 = {
                "n": nodes_count_now,
                "t": t,
                "|eh|_Neu": neuman_errors,
                "|eh|_Dir": dirich_errors,

                "||uh||": energy,
                "|uh|": sobol,
                # "q": q_value
            }
            data_frame = pd.DataFrame(data2)
            st.dataframe(data_frame)
            st.plotly_chart(px.line(df, x='x', y='uh', animation_frame='t', range_y=(-1.1, 1.1)))



            # for i in range(self._i):
            #     nodes, K, l = femdiscretizer.do_discretization(nodes_now)
            #     alpha = TridiagonalMatrixAlgorithm(K, l).solve()
            #     iterations.append(i)
            #     nodes_count_now.append(nodes_now)
            #     r_value.append(calculator.r_calculate(nodes, alpha))
            #     neuman_errors.append(errors.neumann_error(nodes, alpha))
            #     j_value.append(calculator.jump_calculate(nodes, alpha))
            #     dirich_errors.append(errors.dirichlet_error(nodes, alpha))
            #     energy.append(np.sum(norms.energy_norm(nodes, alpha)))
            #     sobol.append(np.sum(norms.sobolev_norm(nodes, alpha)))
            #     if i >= 3:
            #         q_value.append(calculator.q_calculate(sobol, i))
            #     else:
            #         q_value.append(None)
            #     if 'true_solution' in st.session_state:
            #         u = st.session_state['true_solution']
            #         a = u(nodes)
            #         u_sobol.append(np.sum(norms.sobolev_norm(nodes, a)))
            #         u_energy.append(np.sum(norms.energy_norm(nodes, a)))
            #     nodes_now *= 2
            #
            #
            # data2 = {
            #     "i": iterations,
            #     "n": nodes_count_now,
            #     "R(uh)": r_value,
            #     "j(uh)": j_value,
            #     "|eh|_Neu": neuman_errors,
            #     "|eh|_Dir": dirich_errors,
            #     "|uh|": sobol,
            #     "||uh||": energy,
            #     "q": q_value
            # }
            # if 'true_solution' in st.session_state:
            #     data2["|u|"] = u_sobol
            #     data2["||u||"] = u_energy
            # data_frame = pd.DataFrame(data2)
            # st.dataframe(data_frame)
            #
            # data = {
            #     "x": nodes,
            #     "uh": alpha
            # }
            # if 'true_solution' in st.session_state:
            #     y_cols = ['uh', 'u']
            #     u = st.session_state['true_solution']
            #     data['u'] = u(data['x'])
            # else:
            #     y_cols = ['uh']
            # chart_data_frame = pd.DataFrame(data)
            # st.plotly_chart(
            #     px.line(chart_data_frame, x='x', y=y_cols, template='plotly_dark', title='Solution u=uh(x)'))

def main():
    st.set_page_config(page_title='MKO')
    st.header("Finite elements method")
    with st.container():
        input_form = st.form(key="input_form")
        left_column, right_column = st.columns(2)
        with input_form:
            data_class = Data()
            st.subheader('Enter coefficients')
            data_class.coefficients_inputs()

            st.subheader('Choose boundary condition types')

            selected = data_class.boundary_selectors(left_column, right_column)
            data_class.boundary_input_form(selected, left_column, right_column)
            is_submitted = st.form_submit_button('Save', type='primary')
            if is_submitted:
                data_class.create_boundary_problem()

        if not 'BoundaryProblem' in st.session_state:
            st.info('Submit Boundary problem first.')
            return
        else:
            data_class.FEM()
        if not 'discretizer' in st.session_state:
            st.info('Submit data first.')
            return
        else:
            data_class.iterations()


if __name__ == "__main__":
    main()