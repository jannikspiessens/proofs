use itertools::{izip, Itertools};
use std::cell::RefCell;

use feanor_math::homomorphism::{Homomorphism, CanHomFrom};
use feanor_math::integer::{BigIntRing, BigIntRingBase, IntegerRingStore};
use feanor_math::ring::{ RingStore, El };
use feanor_math::rings::{
    zn::{ZnRing, zn_big::Zn, ZnRingStore},
    finite::FiniteRing
};

use crate::{
    FSRng,
    commit::abdlop::{ZZbig, ABDLOP, ABDLOPcommitment, ABDLOPmessage, ABDLOPopening, ABDLOPparts},
    lattice::{gen_infbnd, gen_vector_dgauss, gen_vector_latrejsampl, inner_prod, RejSamplModes},
    util::{
        FiatShamirSim,
        matmul::{MatrixMul, SparseMatrixMul, DenseMatrixMul},
    }
};


pub struct LatSigmaProof<R>
    where R: RingStore
{
    z1: Option<Vec<El<R>>>,
    z2: Vec<El<R>>,
    w: Vec<El<R>>,
    vneg: Option<Vec<El<R>>>,
    fscnt: usize
}


pub struct LatSigmaLinRel<'a, R, MM1, MMm>
    where R: RingStore, MM1: MatrixMul<R = R>, MMm: MatrixMul<R = R>
{
    R1: Option<MM1>,
    Rm: Option<MMm>,
    RmB: Option<DenseMatrixMul<'a, R>>,
    u: Vec<El<R>>
}

// TODO: add matrix-matrix mul to MatrixMul trait
// then make this impl generic in MMm
impl<'a, R, MM1> LatSigmaLinRel<'a, R, MM1, SparseMatrixMul<'a, R>>
    where R: RingStore, MM1: MatrixMul<R = R>
{
    pub fn new(cs: &'a ABDLOP<'a, R>,
        R1: Option<MM1>, Rm: Option<SparseMatrixMul<'a, R>>, u: Vec<El<R>>) -> Self
    {
        assert!(R1.is_some() || Rm.is_some());
        assert!(Rm.is_none() || cs.has_bdlop());
        assert!(Rm.as_ref().is_none_or(|x| x.columns() == cs.get_B().unwrap().rows()));

        let ring = cs.ring();
        let RmB = Rm.as_ref().map(|x| {
            let B = cs.get_B().unwrap();
            // NOTE: I know this is suboptimal but it's precomp so who cares :)
            let data = x.iter_rows().flat_map(|Rmrow|
                (0..B.columns()).map(|j|
                    Rmrow.iter().fold(ring.zero(), |acc, (k, Rmel)|
                        ring.add(acc, ring.mul_ref(Rmel, B.get(*k, j)))
                    )
                )
            ).collect();
            DenseMatrixMul::new(ring, B.columns(), data, "RmB_precomp")
        });

        Self { R1, Rm, RmB, u }
    }
}


pub type LatSigmaDefault<'a, R> = LatSigma<'a, R, DenseMatrixMul<'a, R>, SparseMatrixMul<'a, R>>;

pub struct LatSigma<'a, R, MM1, MMm>
    where R: RingStore, MM1: MatrixMul<R = R>, MMm: MatrixMul<R = R>
{
    cs: ABDLOP<'a, R>,
    fs: RefCell<FiatShamirSim<FSRng>>,
    gamma: (Option<f64>, f64),
    challbnd: El<BigIntRing>, // TODO: add more general distributions besides inf bounds
    rsmode: RejSamplModes,
    linrel: Option<LatSigmaLinRel<'a, R, MM1, MMm>>
}

impl<'a, R, MM1, MMm> LatSigma<'a, R, MM1, MMm>
    where R: RingStore, MM1: MatrixMul<R = R>, MMm: MatrixMul<R = R>
{
    pub fn ring(&self) -> &R { self.cs.ring() }

    pub fn abdlop(&self) -> &ABDLOP<'a, R> { &self.cs }

    pub fn get_fs(&self) -> &RefCell<FiatShamirSim<FSRng>> { &self.fs }

    pub fn set_linrel(&mut self, linrel: LatSigmaLinRel<'a, R, MM1, MMm>) {
        self.linrel = Some(linrel);
    }

    fn get_sigma(&self, part: ABDLOPparts) -> f64 {
        ZZbig.to_float_approx(&self.challbnd) * (match part {
            ABDLOPparts::Ajtai =>
                self.gamma.0.unwrap() * ZZbig.to_float_approx(self.cs.get_bnd1().as_ref().unwrap()),
            ABDLOPparts::BDLOP =>
                self.gamma.1 * ZZbig.to_float_approx(&self.cs.get_bnd2())
        })
    }

    fn get_zbound(&self, part: ABDLOPparts) -> f64 {
        self.get_sigma(part) * (match part {
            ABDLOPparts::Ajtai => self.cs.get_m1().unwrap(),
            ABDLOPparts::BDLOP => self.cs.get_m2()
        } as f64) * 2f64.sqrt()
    }
}

impl<'a, R, MM1, MMm> LatSigma<'a, R, MM1, MMm>
    where R: RingStore<
        Type: FiniteRing + CanHomFrom<BigIntRingBase> + ZnRing>,
        MM1: MatrixMul<R = R>, MMm: MatrixMul<R = R>
{
    pub fn new(cs: ABDLOP<'a, R>,
        gamma: (Option<f64>, f64), challbnd: El<BigIntRing>, 
        rsmode: RejSamplModes
    ) -> Self {
        assert!(!cs.has_ajtai() || gamma.0.is_some());
        let fs = RefCell::new(FiatShamirSim::<FSRng>::new());
        Self { cs, fs, gamma, challbnd, rsmode, linrel: None }
    }

    pub fn prove(&self, op: &ABDLOPopening<R>, mes: &ABDLOPmessage<R>) -> LatSigmaProof<R>
    {
        assert!(!self.cs.has_ajtai() || mes.s1().is_some());

        let fsclone = self.fs.borrow().clone();

        let mut rng = self.cs.rng().borrow_mut();
        let y2 = gen_vector_dgauss(self.ring(), &mut rng,
            self.get_sigma(ABDLOPparts::BDLOP), self.cs.get_A2().columns());
        let (w, vneg, y1) = {
            let tmpw = self.cs.get_A2().mulit(&y2);
            let tmpv = (&self.linrel).as_ref().map(|x| x.RmB.as_ref().map(|xx| xx.mulit(&y2)));
            if self.cs.has_ajtai() {
                let y1 = gen_vector_dgauss(self.ring(), &mut rng,
                    self.get_sigma(ABDLOPparts::Ajtai), self.cs.get_A1().unwrap().columns());
                let resw = tmpw.zip(self.cs.get_A1().unwrap().mulit(&y1)).map(|(l, r)|
                    self.ring().add(l, r)).collect_vec();
                let resv = tmpv.map(|x| x.map_or_else(
                    || self.linrel.as_ref().unwrap().R1.as_ref().unwrap().mulit(&y1).map(|el|
                        self.ring().negate(el)).collect_vec(),
                    // NOTE: cannot use map_or here since compiler needs to understand that EITHER
                    // RmBy2 moves directly to vec or is first mapped
                    |RmBy2| if let Some(R1ref) = self.linrel.as_ref().unwrap().R1.as_ref() {
                        R1ref.mulit(&y1).zip(RmBy2).map(|(l, r)|
                            self.ring().sub(r, l)).collect_vec()
                    } else {
                        RmBy2.collect_vec()
                    }
                ));
                (resw, resv, Some(y1))
            } else {
                (tmpw.collect_vec(), tmpv.map(|x| x.unwrap().collect_vec()), None)
            }
        };
        let (z1, z2, fscnt) = if self.cs.has_ajtai() {
            let mut fsmut = self.fs.borrow_mut();
            let (zt, fscnt) = gen_vector_latrejsampl(self.ring(), &mut rng, &mut fsmut,
                &self.challbnd, [self.gamma.0.unwrap(), self.gamma.1],
                [self.get_sigma(ABDLOPparts::Ajtai), self.get_sigma(ABDLOPparts::BDLOP)],
                self.rsmode, [y1.as_ref().unwrap(), &y2], [mes.s1().unwrap(), op]);
            let (z1, z2) = zt.into();
            (Some(z1), z2, fscnt)
        } else {
            let mut fsmut = self.fs.borrow_mut();
            let (zt, fscnt) = gen_vector_latrejsampl(self.ring(), &mut rng, &mut fsmut,
                &self.challbnd, [self.gamma.1], [self.get_sigma(ABDLOPparts::BDLOP)],
                self.rsmode, [&y2], [op]);
            let (z2,) = zt.into();
            (None, z2, fscnt)
        };

        self.fs.replace(fsclone);

        LatSigmaProof{ z1, z2, w, vneg, fscnt }
    }

    pub fn verify(&self, com: &ABDLOPcommitment<R>, proof: &LatSigmaProof<R>) -> bool {
        if !(proof.z1.is_some() == self.cs.has_ajtai() && com.len() == self.cs.comlen())
            { return false };
        
        let ring = self.ring();
        let intring = ring.integer_ring();
        if let Some(z1) = proof.z1.as_ref() {
            if intring.to_float_approx(&inner_prod(ring, &intring, z1.iter(), z1.iter())) >
                self.get_zbound(ABDLOPparts::Ajtai) { return false }
        }
        if intring.to_float_approx(&inner_prod(ring, &intring, proof.z2.iter(), proof.z2.iter())) >
            self.get_zbound(ABDLOPparts::BDLOP) { return false }
        
        let mut fs = self.fs.borrow_mut();
        let Rbnd = Zn::new(ZZbig, ZZbig.clone_el(&self.challbnd));
        let hom = ring.can_hom(&ZZbig).unwrap();
        let mut chall = ring.zero();
        (0..proof.fscnt).for_each(|_| chall = gen_infbnd(fs.get_rng(), &Rbnd, &hom));

        let A2z2iter = self.cs.get_A2().mulit(&proof.z2);
        let lhsiter = if let Some(z1) = proof.z1.as_ref() {
            Box::new(A2z2iter.zip(self.cs.get_A1().unwrap().mulit(z1)).map(|(A2z2i, A1z1i)|
                ring.add(A2z2i, A1z1i))) as Box<dyn Iterator<Item = El<R>>>
        } else { Box::new(A2z2iter) };

        let m1 = self.cs.get_A2().rows();
        if izip!(lhsiter, &proof.w, &com[..m1]).any(|(lhsi, wi, ci)|
            !ring.eq_el(&lhsi, &ring.add_ref_fst(wi, ring.mul_ref(&chall, ci)))) { return false }

        if let Some(linrel) = &self.linrel {
            let challuiter = linrel.u.iter().map(|ui| ring.mul_ref(&chall, ui));
            let lhsiter = if let Some(R1) = linrel.R1.as_ref() {
                Box::new(challuiter.zip(R1.mulit(proof.z1.as_ref().unwrap())).map(|(challui, R1z1i)|
                    ring.add(challui, R1z1i))) as Box<dyn Iterator<Item = El<R>>>
            } else { Box::new(challuiter) };
            let lhsiter2 = if let Some(Rm) = linrel.Rm.as_ref() { 
                let RmBz2iter = linrel.RmB.as_ref().unwrap().mulit(&proof.z2);
                Box::new(izip!(lhsiter, RmBz2iter, Rm.mulit(&com[m1..])).map(|(lhsi, RmBz2i, Rmti)|
                    ring.sub(ring.add(lhsi, RmBz2i), ring.mul_ref_snd(Rmti, &chall))
                )) as Box<dyn Iterator<Item = El<R>>> } else { lhsiter };

            if lhsiter2.zip(proof.vneg.as_ref().unwrap()).any(|(lhsi, vnegi)|
                !ring.eq_el(&lhsi, vnegi)) { return false }
        }
        return true
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use feanor_math::rings::zn::zn_64::Zn;
    use feanor_math::rings::field::AsField;

    use crate::{
        util::gen_random,
        commit::abdlop::ABDLOPmessage,
    };

    type FieldImpl = AsField<Zn>;

    #[test]
    fn test_latsigma() {

        let field = Zn::new(65537).as_field().ok().unwrap();
        // let mut rng = rand::rng();
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        
        let n = 1 << 12;
        let l = 3000;
        let m2 = 700; // NOTE: should be larger than 640

        let inthom = ZZbig.int_hom();
        let bnd2 = inthom.map(1 << 10);
        
        let m1 = None;
        let s1 = None;
        let bnd1 = None;
        // let m1 = Some(200);
        // let bnd1_ = inthom.map(1 << 10);
        // let s1 = Some(gen_vector_infbnd(&field, &mut rng, &bnd1_, m2));
        // let bnd1 = Some(bnd1_);
        
        let m = Some(gen_random(&field, &mut rng, l));
        let mes = ABDLOPmessage::new(&s1, &m);
        
        let abdlop = ABDLOP::random(&field, rng, n, Some(l), m1, m2, bnd1, bnd2);
        let (com, op) = abdlop.commit(&mes);

        assert!(abdlop.open(&com, &mes, &op));

        // NOTE: we set gamma here to ensure M\approx 3
        let rsmode = RejSamplModes::Mode1;
        let gamma1 = Some(13f64);
        let gamma2 = 13f64;
        // let rsmode = RejSamplModes::Mode2;
        // let gamma1 = Some(0.675f64);
        // let gamma2 = 0.675f64;
        let gamma = (gamma1, gamma2);
        let challbnd = ZZbig.power_of_two(128);

        let latsigma: LatSigmaDefault<FieldImpl>
            = LatSigma::new(abdlop, gamma, challbnd, rsmode);

        let proof = latsigma.prove(&op, &mes);

        assert!(latsigma.verify(&com, &proof));
    }
}
