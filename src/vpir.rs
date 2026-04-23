use itertools::Itertools;
use std::cell::{RefCell};

use feanor_math::integer::{BigIntRing, IntegerRingStore};
use feanor_math::divisibility::DivisibilityRing;
use feanor_math::rings::field::AsField;
use feanor_math::field::Field;
use feanor_math::ring::{RingStore, El};
use feanor_math::rings::finite::{FiniteRing, FiniteRingStore};

use crate::{
    codes::foldablecodes::RSFoldableCode,
    basefold::{
        MultilinearPCS,
        BaseFoldPCS
    },
    util::{gen_vector, Coeff, CoeffRing},
    multilinear::{
        sumcheck::{Sumcheck, SumcheckBase},
        MultilinearBasisEvals,
        evals_to_coeffs_inplace, evaluate_at_fromcoeff,
        evaluate_at_fromevals
    },
    matmul::{MatrixMul, DenseMatrixMul}
};

pub struct vPIRPIOP<'a, PCS: MultilinearPCS>
{
    vc_rows: usize,
    vc_cols: usize,
    tau: Vec<Coeff<PCS::Poly>>,
    z: RefCell<Vec<Coeff<PCS::Poly>>>,
    M: DenseMatrixMul<'a, CoeffRing<PCS::Poly>>,
    pcs: PCS,
    zcoeff: Vec<Coeff<PCS::Poly>>,
    com: PCS::C,
    zMtau: Coeff<PCS::Poly>
}

impl<'a, R> vPIRPIOP<'a, BaseFoldPCS<'a, AsField<R>, RSFoldableCode<'a, AsField<R>>>>
    where R: RingStore + Clone, R::Type: DivisibilityRing + FiniteRing,
{
    pub fn new_extra(field: &'a AsField<R>, z: Vec<El<AsField<R>>>, zM: Vec<El<AsField<R>>>, M: DenseMatrixMul<'a, AsField<R>>, ver_rep: usize)
        -> Self
    {
        assert!(z.len().is_power_of_two());
        let vc_cols = z.len().ilog2() as usize;
        let mut zcoeff = z.iter().map(|el| field.clone_el(el)).collect_vec();
        evals_to_coeffs_inplace(field, vc_cols, &mut zcoeff);
        // TODO: set reasonable parameters here
        let k0 = 1;
        let c = 8;
        let pcs = BaseFoldPCS::new(field, vc_cols, k0, c, ver_rep);
        let com = pcs.commit(&zcoeff);

        let vc_rows = M.rows().ilog2() as usize;

        let tau = (0..vc_rows).map(|_| pcs.get_challenge()).collect_vec();

        let zMtau = evaluate_at_fromevals(field, &tau, &zM);

        Self {
            vc_rows,
            vc_cols,
            tau,
            z: RefCell::new(z),
            M,
            pcs,
            zcoeff,
            com,
            zMtau
        }
    }

    pub fn new(ring: &'a AsField<R>, z: Vec<El<AsField<R>>>, M: DenseMatrixMul<'a, AsField<R>>,
        ver_rep: usize) -> Self
    {
        let zM = M.mul(&z);
        vPIRPIOP::new_extra(ring, z, zM, M, ver_rep)
    }

    pub fn proofsize(&self) -> usize {
        let rowchecksize = 4*self.vc_rows;
        let linchecksize = 3*self.vc_cols;
        let ZZbig: BigIntRing = BigIntRing::RING;
        let bits = ZZbig.abs_log2_ceil(&self.pcs.coeffring().characteristic(ZZbig).unwrap()).unwrap();
        return self.pcs.proofsize() + bits*(rowchecksize + linchecksize)
    }

    pub fn matrix(&self) -> &DenseMatrixMul<'a, AsField<R>> {
        &self.M
    }
}
    
impl<'a, PCS: MultilinearPCS> vPIRPIOP<'a, PCS>
{
    pub fn field(&self) -> &CoeffRing<PCS::Poly> {
        self.pcs.coeffring()
    }

    pub fn varcount_rows(&self) -> usize {
        self.vc_rows
    }

    pub fn varcount_cols(&self) -> usize {
        self.vc_cols
    }

    pub fn tau(&self) -> &Vec<Coeff<PCS::Poly>> {
        &self.tau
    }

    pub fn challenge(&mut self) -> Coeff<PCS::Poly> {
        self.pcs.get_challenge()
    }

    pub fn get_zM(&self) -> [&RefCell<Vec<Coeff<PCS::Poly>>>; 1] {
        [&self.z]
    }

    pub fn zMtau(&self) -> &Coeff<PCS::Poly> {
        &self.zMtau
    }
}

use rand::RngCore;
use rand_seeder::{Seeder, SipRng};

impl<'a, R> vPIRPIOP<'a, BaseFoldPCS<'a, AsField<R>, RSFoldableCode<'a, AsField<R>>>>
    where R: RingStore + Clone, R::Type: DivisibilityRing + FiniteRing,
{
    pub fn random(field: &'a AsField<R>, vc: usize, vcrows: usize, ver_rep: usize) -> Self {
        let mut rng: SipRng = Seeder::from("vPIRPIOP").into_rng();
        let mut z = gen_vector::<El<AsField<R>>>(||
            // field.random_element(rand::random::<u64>), 1 << vc);
            field.random_element(|| rng.next_u64()), 1 << vc);
        while z.iter().all(|zi| field.is_zero(zi)) {
            // z = gen_vector::<El<F>>(|| field.random_element(rand::random::<u64>), 1 << vc);
            z = gen_vector::<El<AsField<R>>>(|| field.random_element(|| rng.next_u64()), 1 << vc);
        }
        // let (r1cs, zA, zB, zC) = R1CS::random_from(field, &z, 1 << vcrows);
        let Mdata = gen_vector::<El<AsField<R>>>(||
            field.random_element(|| rng.next_u64()), (1 << vcrows)*z.len());

        let M = DenseMatrixMul::new(field, z.len(), Mdata,
            format!("PIRdatabase{vc}{vcrows}").as_str());
        let zM = M.mul(&z);
        vPIRPIOP::new_extra(field, z, zM, M, ver_rep)
    }
}

impl<'a, PCS> vPIRPIOP<'a, PCS>
    where PCS: MultilinearPCS, CoeffRing<PCS::Poly>: RingStore<Type: Field + FiniteRing>
{
    pub fn execute(self) -> bool {
        let linchecksum = self.field().clone_el(&self.zMtau);
        let lincheck = vPIRLincheck::for_piop(self);
        if let Some(rX) = lincheck.execute(linchecksum) {
            println!("SpartanLincheck complete!");
            lincheck.check_eval(rX)
        } else { false }
    }
}


pub struct vPIRLincheckBase<'a, PCS: MultilinearPCS>
{
    piop: vPIRPIOP<'a, PCS>
}

impl<'a, PCS:MultilinearPCS> vPIRLincheckBase<'a, PCS>
{
    pub fn for_piop(piop: vPIRPIOP<'a, PCS>) -> Self
    {
        Self { piop }
    }
}

impl<'a, PCS> SumcheckBase<2> for vPIRLincheckBase<'a, PCS>
    where PCS: MultilinearPCS, CoeffRing<PCS::Poly>: RingStore<Type: Field + FiniteRing>
{
    type F = CoeffRing<PCS::Poly>;

    fn field(&self) -> &Self::F {
        self.piop.field()
    }

    fn varcount(&self) -> usize {
        self.piop.varcount_cols()
    }

    fn get_challenge(&self) -> El<Self::F> {
        self.piop.pcs.get_challenge()
    }

    fn get_other_eval_points(&self) -> [i32; 1] {
        [-1]
    }

    fn get_scalars<'c, 'd>(&'d self, challs: &'c [El<Self::F>])
        -> impl Iterator<Item = (El<Self::F>, El<Self::F>)> + 'c
        where 'd: 'c
    {
        let field = self.piop.field();
        (0..(1 << (self.varcount() - challs.len()).saturating_sub(1))).map(|_|
            (field.one(), field.one()))
    }
}

pub struct vPIRLincheck<'a, PCS: MultilinearPCS>
{
    base: vPIRLincheckBase<'a, PCS>,
    wsM: RefCell<Vec<Coeff<PCS::Poly>>>,
}

impl<'a, PCS> vPIRLincheck<'a, PCS>
    where PCS: MultilinearPCS, CoeffRing<PCS::Poly>: RingStore<Type: Field>
{

    pub fn for_piop(piop: vPIRPIOP<'a, PCS>) -> Self
    {
        let field = piop.field();
        let M = &piop.M;
        let mut tauM = (0..M.columns()).map(|_| field.zero()).collect_vec();

        let eqevals = MultilinearBasisEvals::new(field, &piop.tau);
        
        M.data().chunks_exact(M.columns()).zip(eqevals).for_each(|(Mrow, eqi)|
            Mrow.iter().zip(tauM.iter_mut()).for_each(|(Mj, rj)| {
                field.add_assign(rj, field.mul_ref_fst(Mj, field.clone_el(&eqi)));
            })
        );

        let wsM = RefCell::new(tauM);
        let base = vPIRLincheckBase::for_piop(piop);
        Self { base, wsM }
    }

    pub fn get_wsM(&self) -> &RefCell<Vec<Coeff<PCS::Poly>>> {
        &self.wsM
    }
}

impl<'a, PCS> Sumcheck<2, 2> for vPIRLincheck<'a, PCS>
    where PCS: MultilinearPCS, CoeffRing<PCS::Poly>: RingStore<Type: Field + FiniteRing>
{
    type SCB = vPIRLincheckBase<'a, PCS>;

    fn get_base(&self) -> &Self::SCB {
        &self.base
    }

    fn get_workspace(&self) -> [&RefCell<Vec<Coeff<PCS::Poly>>>; 2] {
        [&self.wsM, &self.base.piop.z]
    }

    fn compute_term(ring: &CoeffRing<PCS::Poly>, at: [&Coeff<PCS::Poly>; 2], _scalar: &Coeff<PCS::Poly>) -> Coeff<PCS::Poly> {
        ring.mul_ref(at[0], at[1])
    }

    fn check_eval(self, rX: Vec<Coeff<PCS::Poly>>) -> bool {
        // TODO: check _evalM
        let ring = self.base.field();
        let y = {
            let [_evalM, evalz] = self.get_workspace();
            let evalzref = evalz.borrow();
            debug_assert!(evalzref.len() == 1);
            ring.clone_el(&evalzref[0])
        };
        {
            let ev = evaluate_at_fromcoeff(ring, self.base.varcount(), &rX, &self.base.piop.zcoeff);
            debug_assert!(ring.eq_el(&y, &ev[0]));
        }
        let proof = self.base.piop.pcs.eval_fast(&self.base.piop.com, &rX,
            ring.clone_el(&y), &self.base.piop.zcoeff);
        let clonedy = ring.clone_el(&y);
        self.base.piop.pcs.verify(self.base.piop.com, &rX, clonedy, &self.base.piop.zcoeff, proof)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use feanor_math::rings::zn::ZnRingStore;
    use feanor_math::rings::zn::zn_64::Zn;

    const VREP: usize = 100;

    #[test]
    fn test_vpir() {

        let field = Zn::new(65537).as_field().ok().unwrap();

        // let N = 4;
        let N = 12;
        
        let vpir = vPIRPIOP::random(&field, N, N+1, VREP);

        assert!(vpir.execute());
    }
}
