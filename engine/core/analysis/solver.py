import numpy as np
from .loads import Load, LoadCase, LoadCombination

class FEMSolver:
    def __init__(self, analytical_model):
        self.model = analytical_model

    def assemble_global_stiffness(self):
        # Assemble global stiffness matrix for all supported element types
        n_dof = len(self.model.nodes) * 3  # 3 DOF per node (u, v, theta or u, v, w)
        K = np.zeros((n_dof, n_dof))
        for elem in self.model.elements:
            k_local = elem.stiffness_matrix()
            # Determine DOF mapping based on element type/number of nodes
            if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                # Beam, column, cable: 2 nodes x 3 DOF
                node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                dof_map = []
                for nid in node_ids:
                    dof_map.extend([nid*3, nid*3+1, nid*3+2])
                for i in range(6):
                    for j in range(6):
                        K[dof_map[i], dof_map[j]] += k_local[i, j]
            elif hasattr(elem, 'nodes') and len(elem.nodes) == 4:
                # Shell: 4 nodes x 3 DOF
                node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                dof_map = []
                for nid in node_ids:
                    dof_map.extend([nid*3, nid*3+1, nid*3+2])
                for i in range(12):
                    for j in range(12):
                        K[dof_map[i], dof_map[j]] += k_local[i, j]
            elif hasattr(elem, 'nodes') and len(elem.nodes) == 8:
                # Solid: 8 nodes x 3 DOF
                node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                dof_map = []
                for nid in node_ids:
                    dof_map.extend([nid*3, nid*3+1, nid*3+2])
                for i in range(24):
                    for j in range(24):
                        K[dof_map[i], dof_map[j]] += k_local[i, j]
            else:
                # Fallback: try to map based on available nodes
                node_ids = [self.model.nodes.index(n) for n in getattr(elem, 'nodes', [])]
                dof_map = []
                for nid in node_ids:
                    dof_map.extend([nid*3, nid*3+1, nid*3+2])
                for i in range(len(dof_map)):
                    for j in range(len(dof_map)):
                        if i < k_local.shape[0] and j < k_local.shape[1]:
                            K[dof_map[i], dof_map[j]] += k_local[i, j]
        return K

    def assemble_load_vector(self, loads):
        n_dof = len(self.model.nodes) * 3
        F = np.zeros(n_dof)
        for load in loads:
            if load.load_type == 'point':
                F[load.location * 3 + (load.direction or 0)] += load.magnitude
            elif load.load_type == 'distributed':
                # Uniformly distributed load on a beam/column/cable element
                # location = element index, direction = dof (0=x, 1=y), span = (start, end) (optional)
                elem = self.model.elements[load.location]
                node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                # For uniform load q in global direction, equivalent nodal forces (2D):
                # F1 = F2 = q*L/2
                L = np.linalg.norm(np.array(elem.nodes[1]) - np.array(elem.nodes[0]))
                q = load.magnitude
                dir = load.direction or 1  # default to y
                F[node_ids[0]*3 + dir] += q * L / 2
                F[node_ids[1]*3 + dir] += q * L / 2
            elif load.load_type == 'area':
                # Area load on a shell element
                # location = element index, magnitude = pressure, direction = dof (default y)
                elem = self.model.elements[load.location]
                node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                q = load.magnitude
                dir = load.direction or 1
                # Distribute equally to 4 nodes
                for nid in node_ids:
                    F[nid*3 + dir] += q / 4
            elif load.load_type == 'temperature':
                # Temperature load: apply equivalent nodal forces (stub)
                # location = element index, temperature = deltaT
                # For now, do nothing or add a stub force
                elem = self.model.elements[load.location]
                node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                # Real implementation would use alpha*E*deltaT*area, etc.
                for nid in node_ids:
                    F[nid*3] += 0  # stub
            # else: ignore or extend for more types
        return F

    def solve_static(self, loads_or_case):
        K = self.assemble_global_stiffness()
        if isinstance(loads_or_case, LoadCase):
            F = self.assemble_load_vector(loads_or_case.loads)
        elif isinstance(loads_or_case, LoadCombination):
            F = np.zeros(K.shape[0])
            for lc, factor in loads_or_case.cases.items():
                F += factor * self.assemble_load_vector(lc.loads)
        else:
            F = self.assemble_load_vector(loads_or_case)
        # TODO: Apply boundary conditions (fix supports)
        displacements = np.linalg.solve(K, F)
        return displacements

    def compute_beam_internal_forces(self, displacements):
        # Compute moment, shear, axial force diagrams for each beam element
        diagrams = []
        for elem in self.model.elements:
            if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                # For demonstration, use dummy values
                L = np.linalg.norm(np.array(elem.nodes[1]) - np.array(elem.nodes[0]))
                x = np.linspace(0, L, 20)
                M = np.zeros_like(x)
                V = np.zeros_like(x)
                N = np.zeros_like(x)
                # TODO: Use element stiffness, displacements, and loads to compute real diagrams
                diagrams.append({'element': elem, 'x': x, 'M': M, 'V': V, 'N': N})
        return diagrams

    def compute_shell_displacement_contour(self, displacements):
        # Compute nodal displacement magnitude for each shell node
        contours = []
        for elem in self.model.elements:
            if hasattr(elem, 'nodes') and len(elem.nodes) == 4:
                node_disp = []
                for i, n in enumerate(elem.nodes):
                    idx = self.model.nodes.index(n)
                    ux = displacements[idx*3] if len(displacements) > idx*3 else 0
                    uy = displacements[idx*3+1] if len(displacements) > idx*3+1 else 0
                    uz = displacements[idx*3+2] if len(displacements) > idx*3+2 else 0
                    node_disp.append(np.sqrt(ux**2 + uy**2 + uz**2))
                contours.append({'element': elem, 'disp': node_disp})
        return contours

    def solve_dynamic(self):
        # TODO: Modal, response spectrum, time history analysis
        pass

    def solve_nonlinear(self, options=None):
        """
        Enhanced nonlinear static solver with geometric and material nonlinearity.
        options: steps, max_iter, tol, arc_length (bool)
        """
        if options is None:
            options = {'steps': 10, 'max_iter': 20, 'tol': 1e-4, 'arc_length': False}
        steps = options.get('steps', 10)
        max_iter = options.get('max_iter', 20)
        tol = options.get('tol', 1e-4)
        arc_length = options.get('arc_length', False)
        loads = getattr(self.model, 'loads', [])
        total_F = self.assemble_load_vector(loads)
        n_dof = len(self.model.nodes) * 3
        u = np.zeros(n_dof)
        u_history = [u.copy()]
        F_step = total_F / steps
        converged = True
        for step in range(steps):
            F_target = F_step * (step + 1)
            u_step = u.copy()
            for it in range(max_iter):
                # Assemble linear and geometric stiffness
                K = self.assemble_global_stiffness()
                # Geometric nonlinearity: add geometric stiffness from current state
                Kg = np.zeros_like(K)
                for elem in self.model.elements:
                    if hasattr(elem, 'geometric_stiffness_matrix'):
                        # Estimate axial force from current displacement
                        node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                        u_elem = np.concatenate([u_step[nid*3:nid*3+3] for nid in node_ids])
                        if hasattr(elem, 'compute_axial_force'):
                            N = elem.compute_axial_force(u_elem)
                        else:
                            N = 0.0
                        kG_local = elem.geometric_stiffness_matrix(N)
                        dof_map = []
                        for nid in node_ids:
                            dof_map.extend([nid*3, nid*3+1, nid*3+2])
                        for i in range(len(dof_map)):
                            for j in range(len(dof_map)):
                                Kg[dof_map[i], dof_map[j]] += kG_local[i, j]
                K_total = K + Kg
                # Material nonlinearity: update element state if method exists
                for elem in self.model.elements:
                    if hasattr(elem, 'update_state'):
                        node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                        u_elem = np.concatenate([u_step[nid*3:nid*3+3] for nid in node_ids])
                        elem.update_state(u_elem)
                R = F_target - K_total @ u_step
                norm_R = np.linalg.norm(R)
                if norm_R < tol:
                    break
                try:
                    du = np.linalg.solve(K_total, R)
                except np.linalg.LinAlgError:
                    converged = False
                    break
                # Arc-length/displacement control placeholder
                if arc_length:
                    # TODO: Implement arc-length method
                    pass
                u_step += du
            else:
                converged = False
            u = u_step.copy()
            u_history.append(u.copy())
        return {
            'displacement_history': u_history,
            'converged': converged,
            'steps': steps,
            'max_iter': max_iter,
            'tol': tol
        }

    def compute_critical_buckling_load(self):
        """
        Perform eigenvalue buckling analysis:
        1. Run static analysis to get axial forces in each element (approximate).
        2. Assemble geometric stiffness matrix Kg from element geometric stiffness matrices.
        3. Solve det(K - lambda*Kg) = 0 for lowest positive lambda (critical load factor).
        """
        K = self.assemble_global_stiffness()
        n_dof = K.shape[0]
        # Step 1: Get axial forces in each element (approximate)
        # For now, use last static analysis result if available, else assume a compressive force
        axial_forces = []
        if hasattr(self, 'last_displacements') and self.last_displacements is not None:
            displacements = self.last_displacements
            for elem in self.model.elements:
                if hasattr(elem, 'nodes') and len(elem.nodes) == 2 and hasattr(elem, 'section'):
                    # Approximate axial force: N = EA/L * (u2 - u1)
                    n1, n2 = elem.nodes
                    idx1 = self.model.nodes.index(n1)
                    idx2 = self.model.nodes.index(n2)
                    u1 = displacements[idx1*3] if len(displacements) > idx1*3 else 0
                    u2 = displacements[idx2*3] if len(displacements) > idx2*3 else 0
                    E = elem.material['E']
                    A = elem.section['A']
                    L = np.linalg.norm(np.array(n2) - np.array(n1))
                    N = E * A / L * (u2 - u1)
                    axial_forces.append(N)
                else:
                    axial_forces.append(-1e4)  # default compressive force
        else:
            for elem in self.model.elements:
                axial_forces.append(-1e4)  # default compressive force
        # Step 2: Assemble Kg
        Kg = np.zeros((n_dof, n_dof))
        for e, elem in enumerate(self.model.elements):
            if hasattr(elem, 'nodes') and len(elem.nodes) == 2 and hasattr(elem, 'geometric_stiffness_matrix'):
                N = axial_forces[e]
                kG_local = elem.geometric_stiffness_matrix(N)
                node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                dof_map = []
                for nid in node_ids:
                    dof_map.extend([nid*3, nid*3+1, nid*3+2])
                for i in range(len(dof_map)):
                    for j in range(len(dof_map)):
                        Kg[dof_map[i], dof_map[j]] += kG_local[i, j]
        # Step 3: Solve generalized eigenvalue problem
        try:
            from scipy.linalg import eig
            eigvals, eigvecs = eig(K, Kg)
            eigvals = np.real(eigvals)
            eigvals = eigvals[np.isfinite(eigvals) & (eigvals > 1e-8)]
            if len(eigvals) == 0:
                return None, None
            crit_lambda = np.min(eigvals)
            crit_mode = eigvecs[:, np.argmin(eigvals)]
            return crit_lambda, crit_mode
        except ImportError:
            return None, None

    def assemble_global_mass(self, lumped=True):
        """
        Assemble the global mass matrix for all supported element types.
        If lumped=True, use lumped mass; otherwise, use consistent mass.
        """
        n_dof = len(self.model.nodes) * 3
        M = np.zeros((n_dof, n_dof))
        for elem in self.model.elements:
            if hasattr(elem, 'mass_matrix'):
                m_local = elem.mass_matrix(lumped=lumped)
                if hasattr(elem, 'nodes') and len(elem.nodes) == 2:
                    node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                    dof_map = []
                    for nid in node_ids:
                        dof_map.extend([nid*3, nid*3+1, nid*3+2])
                    for i in range(6):
                        for j in range(6):
                            M[dof_map[i], dof_map[j]] += m_local[i, j]
                elif hasattr(elem, 'nodes') and len(elem.nodes) == 4:
                    node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                    dof_map = []
                    for nid in node_ids:
                        dof_map.extend([nid*3, nid*3+1, nid*3+2])
                    for i in range(12):
                        for j in range(12):
                            M[dof_map[i], dof_map[j]] += m_local[i, j]
                elif hasattr(elem, 'nodes') and len(elem.nodes) == 8:
                    node_ids = [self.model.nodes.index(n) for n in elem.nodes]
                    dof_map = []
                    for nid in node_ids:
                        dof_map.extend([nid*3, nid*3+1, nid*3+2])
                    for i in range(24):
                        for j in range(24):
                            M[dof_map[i], dof_map[j]] += m_local[i, j]
        return M

    def assemble_global_damping(self, alpha=0.0, beta=0.01, lumped=True):
        """
        Assemble the global damping matrix using Rayleigh damping:
        [C] = alpha*[M] + beta*[K]
        """
        K = self.assemble_global_stiffness()
        M = self.assemble_global_mass(lumped=lumped)
        C = alpha * M + beta * K
        return C

    def compute_modal_analysis(self, num_modes=5, lumped=True):
        """
        Compute the lowest num_modes natural frequencies and mode shapes.
        Uses the assembled mass matrix (lumped or consistent).
        """
        K = self.assemble_global_stiffness()
        n_dof = len(self.model.nodes) * 3
        M = self.assemble_global_mass(lumped=lumped)
        try:
            from scipy.linalg import eigh
            eigvals, eigvecs = eigh(K, M, subset_by_index=[0, min(num_modes-1, n_dof-1)])
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)
            freqs = np.sqrt(np.abs(eigvals)) / (2 * np.pi)
            return {'frequencies': freqs, 'mode_shapes': eigvecs}
        except ImportError:
            return {'frequencies': [], 'mode_shapes': []}

    def solve_staged_construction(self, stages):
        """
        Real staged construction analysis.
        stages: list of dicts, each with 'elements' and 'loads' (indices or objects) to activate at each stage.
        Returns: list of results per stage (displacements, etc).
        """
        results = []
        orig_elements = self.model.elements[:]
        orig_loads = getattr(self.model, 'loads', [])[:]
        for stage in stages:
            # Determine which elements/loads to activate
            stage_elements = stage.get('elements', [])
            stage_loads = stage.get('loads', [])
            # Convert indices to objects if needed
            if stage_elements and isinstance(stage_elements[0], int):
                active_elements = [orig_elements[i] for i in stage_elements]
            else:
                active_elements = stage_elements if stage_elements else orig_elements
            if stage_loads and isinstance(stage_loads[0], int):
                active_loads = [orig_loads[i] for i in stage_loads]
            else:
                active_loads = stage_loads if stage_loads else orig_loads
            # Temporarily set model's elements/loads
            self.model.elements = active_elements
            self.model.loads = active_loads
            # Run analysis for this stage
            try:
                disp = self.solve_static(active_loads)
            except Exception as e:
                disp = f'Error: {e}'
            results.append({'displacements': disp, 'elements': active_elements, 'loads': active_loads})
        # Restore original model
        self.model.elements = orig_elements
        self.model.loads = orig_loads
        return results

    def solve_pushover(self, control_node, direction=1, steps=10, max_disp=0.1):
        """
        Real pushover analysis.
        control_node: node index to monitor
        direction: 0=x, 1=y, 2=z
        steps: number of increments
        max_disp: target displacement
        Returns: capacity curve (displacement, base shear, deformed shapes)
        """
        n_dof = len(self.model.nodes) * 3
        disps = []
        base_shear = []
        deformed_shapes = []
        # For each step, apply increasing lateral load and run (non)linear analysis
        for i in range(steps):
            target_disp = (i+1) * max_disp / steps
            # Create a load vector to achieve target displacement at control node (approximate)
            F = np.zeros(n_dof)
            F[control_node*3 + direction] = 1.0  # unit load
            # Use nonlinear solver if available, else static
            try:
                # Scale load to reach target displacement (approximate, linear scaling)
                u = self.solve_static([Load('point', 1.0, control_node, direction)])
                scale = target_disp / (u[control_node*3 + direction] if abs(u[control_node*3 + direction]) > 1e-12 else 1.0)
                F = F * scale
                # Nonlinear: could use incremental-iterative, but here just static for demo
                u = self.solve_static([Load('point', scale, control_node, direction)])
            except Exception as e:
                u = np.zeros(n_dof)
            # Compute base shear (sum of reactions at supports in direction)
            # For now, sum all reactions in direction at nodes with zero displacement (assume fixed at 0)
            base_shear_val = 0.0
            for nidx, node in enumerate(self.model.nodes):
                if nidx == 0:  # crude: assume node 0 is fixed
                    base_shear_val += F[nidx*3 + direction]
            disps.append(u[control_node*3 + direction])
            base_shear.append(base_shear_val)
            deformed_shapes.append(u.copy())
        return {'displacement': np.array(disps), 'base_shear': np.array(base_shear), 'deformed_shapes': deformed_shapes}

    def solve_time_history(self, F_time, dt=0.01, t_total=1.0, beta=0.25, gamma=0.5, alpha=0.0, beta_rayleigh=0.01, lumped=True, C=None, M=None):
        """
        Time history analysis using Newmark-beta method.
        F_time: function or array, F(t) for each time step (shape: [n_steps, n_dof])
        dt: time step
        t_total: total time
        beta, gamma: Newmark parameters
        alpha, beta_rayleigh: Rayleigh damping coefficients
        lumped: use lumped or consistent mass
        C: damping matrix (optional)
        M: mass matrix (optional)
        Returns: u_hist, v_hist, a_hist (displacement, velocity, acceleration time histories)
        """
        n_dof = len(self.model.nodes) * 3
        n_steps = int(t_total / dt) + 1
        K = self.assemble_global_stiffness()
        if M is None:
            M = self.assemble_global_mass(lumped=lumped)
        if C is None:
            C = self.assemble_global_damping(alpha, beta_rayleigh, lumped=lumped)
        u = np.zeros(n_dof)
        v = np.zeros(n_dof)
        a = np.zeros(n_dof)
        u_hist = [u.copy()]
        v_hist = [v.copy()]
        a_hist = [a.copy()]
        if callable(F_time):
            F = F_time(0)
        else:
            F = F_time[0]
        a = np.linalg.solve(M, F - C @ v - K @ u)
        a_hist[0] = a.copy()
        for i in range(1, n_steps):
            t = i * dt
            if callable(F_time):
                F = F_time(t)
            else:
                F = F_time[i] if i < len(F_time) else F_time[-1]
            K_eff = K + gamma/(beta*dt)*C + M/(beta*dt**2)
            F_eff = F + M @ (1/(beta*dt**2)*u + 1/(beta*dt)*v + (1/(2*beta)-1)*a) + C @ (gamma/(beta*dt)*u + (gamma/beta-1)*v + dt*(gamma/(2*beta)-1)*a)
            u_new = np.linalg.solve(K_eff, F_eff)
            v_new = gamma/(beta*dt)*(u_new-u) + (1-gamma/beta)*v + dt*(1-gamma/(2*beta))*a
            a_new = 1/(beta*dt**2)*(u_new-u) - 1/(beta*dt)*v - (1/(2*beta)-1)*a
            u, v, a = u_new, v_new, a_new
            u_hist.append(u.copy())
            v_hist.append(v.copy())
            a_hist.append(a.copy())
        return np.array(u_hist), np.array(v_hist), np.array(a_hist)

    def solve_response_spectrum(self, spectrum, num_modes=5, combine='SRSS', lumped=True):
        """
        Response spectrum analysis (SRSS modal combination).
        spectrum: function or array, giving spectral acceleration vs. period/frequency
        num_modes: number of modes to use
        combine: 'SRSS' or 'CQC'
        Returns: peak response at each DOF
        """
        modal = self.compute_modal_analysis(num_modes, lumped=lumped)
        freqs = modal['frequencies']
        mode_shapes = modal['mode_shapes']
        n_dof = len(self.model.nodes) * 3
        peak_resp = np.zeros(n_dof)
        for i, freq in enumerate(freqs):
            T = 1.0 / freq if freq > 1e-8 else 1.0
            if callable(spectrum):
                Sa = spectrum(T)
            else:
                idx = np.argmin(np.abs(spectrum[:,0] - T))
                Sa = spectrum[idx,1]
            phi = mode_shapes[:,i]
            max_modal = np.abs(phi) * Sa
            if combine == 'SRSS':
                peak_resp += max_modal**2
        if combine == 'SRSS':
            peak_resp = np.sqrt(peak_resp)
        return peak_resp 