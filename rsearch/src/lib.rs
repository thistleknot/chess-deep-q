//! rsearch — full-width alpha-beta + quiescence over the repo's 809-feature linear eval
//! (spec/rust-search.spec.md). Eval mirrors cem_loop.encode_features EXACTLY: 12x64 piece
//! planes + turn + 40 KnightCap-donor features, same normalizations, same indices.
//! Search: iterative-deepening negamax, full width, captures-until-quiet quiescence with
//! stand-pat, tanh at the leaf (values White-absolute in [-1,1] at the Python boundary).

use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves,
    BitBoard, Board, Color, File, Move, Piece, Rank, Square,
};
use pyo3::prelude::*;

const NFEAT: usize = 809;
const NAMAP: usize = 897; // amap encoder (encoders._amap): [0:769] pst planes+stm,
                          // [769:833] white-attacked squares, [833:897] black-attacked
const TYPE_VAL: [f64; 6] = [1.0, 3.0, 3.0, 5.0, 9.0, 0.0]; // P N B R Q K

fn bb_u64(bb: BitBoard) -> u64 {
    bb.0
}

fn file_mask(f: usize) -> u64 {
    0x0101_0101_0101_0101u64 << f
}

fn adj_files(f: usize) -> u64 {
    let mut m = 0u64;
    if f > 0 {
        m |= file_mask(f - 1);
    }
    if f < 7 {
        m |= file_mask(f + 1);
    }
    m
}

/// front span for passed-pawn test: same+adjacent files, ranks strictly ahead
fn front_span(white: bool, sq: usize) -> u64 {
    let f = sq % 8;
    let r = sq / 8;
    let files = file_mask(f) | adj_files(f);
    let ahead: u64 = if white {
        if r >= 7 {
            0
        } else {
            u64::MAX << (8 * (r + 1))
        }
    } else if r == 0 {
        0
    } else {
        (1u64 << (8 * r)) - 1
    };
    files & ahead
}

struct Eval {
    w: Vec<f64>,   // len NFEAT (kc-809) or NAMAP (amap-897) — mode = length
    bias: f64,
}

impl Eval {
    /// White-absolute pre-tanh linear score; feature semantics == cem_loop.encode_features
    /// (809 weights) or encoders._amap (897 weights — the confirmed feature winner).
    fn score(&self, board: &Board) -> f64 {
        let occ_w = bb_u64(board.colors(Color::White));
        let occ_b = bb_u64(board.colors(Color::Black));
        let occ = [occ_w, occ_b];
        let occ_all = occ_w | occ_b;
        let pieces_bb = [
            bb_u64(board.pieces(Piece::Pawn)),
            bb_u64(board.pieces(Piece::Knight)),
            bb_u64(board.pieces(Piece::Bishop)),
            bb_u64(board.pieces(Piece::Rook)),
            bb_u64(board.pieces(Piece::Queen)),
            bb_u64(board.pieces(Piece::King)),
        ];
        let mut s = self.bias;
        // [0:768] planes: White P..K then Black P..K, feature = plane*64 + sq
        for side in 0..2 {
            for pt in 0..6 {
                let mut m = pieces_bb[pt] & occ[side];
                while m != 0 {
                    let sq = m.trailing_zeros() as usize;
                    s += self.w[(side * 6 + pt) * 64 + sq];
                    m &= m - 1;
                }
            }
        }
        // [768] side to move
        if board.side_to_move() == Color::White {
            s += self.w[768];
        }
        // attack pass: unions + per-piece (sq, type, mask); king in union, not in list
        let mut atk = [0u64, 0u64];
        let mut plist: [Vec<(usize, usize, u64)>; 2] = [Vec::with_capacity(16), Vec::with_capacity(16)];
        for side in 0..2 {
            let color = if side == 0 { Color::White } else { Color::Black };
            let blockers = BitBoard(occ_all);
            let mut a = 0u64;
            for pt in 0..5 {
                let mut m = pieces_bb[pt] & occ[side];
                while m != 0 {
                    let sqi = m.trailing_zeros() as usize;
                    let sq = Square::index(sqi);
                    let am = match pt {
                        0 => bb_u64(get_pawn_attacks(sq, color)),
                        1 => bb_u64(get_knight_moves(sq)),
                        2 => bb_u64(get_bishop_moves(sq, blockers)),
                        3 => bb_u64(get_rook_moves(sq, blockers)),
                        _ => bb_u64(get_bishop_moves(sq, blockers)) | bb_u64(get_rook_moves(sq, blockers)),
                    };
                    a |= am;
                    plist[side].push((sqi, pt, am));
                    m &= m - 1;
                }
            }
            let k = pieces_bb[5] & occ[side];
            if k != 0 {
                a |= bb_u64(get_king_moves(Square::index(k.trailing_zeros() as usize)));
            }
            atk[side] = a;
        }
        if self.w.len() == NAMAP {
            // amap mode: per-square any-attack bits per side — python parity:
            // board.is_attacked_by(color, sq) == union of that side's piece attacks
            for side in 0..2 {
                let mut m = atk[side];
                while m != 0 {
                    let sq = m.trailing_zeros() as usize;
                    s += self.w[769 + 64 * side + sq];
                    m &= m - 1;
                }
            }
            return s;
        }
        for side in 0..2 {
            let opp = 1 - side;
            let own = occ[side];
            let sign = if side == 0 { 1.0 } else { 1.0 }; // weights carry the sign per block
            let _ = sign;
            let own_p = pieces_bb[0] & own;
            let opp_p = pieces_bb[0] & occ[opp];
            let own_r = pieces_bb[3] & own;
            // [769:771] bishop pair
            if (pieces_bb[2] & own).count_ones() >= 2 {
                s += self.w[769 + side];
            }
            // hung
            let hung = own & atk[opp] & !atk[side] & !pieces_bb[5];
            s += self.w[787 + side] * (hung.count_ones() as f64) / 5.0;
            // one pass: mobility, safe mobility, hung value, rook file features
            let mut mob = [0u32; 4];
            let mut smob = [0u32; 4];
            let mut hval = 0.0;
            let mut ropen = 0u32;
            let mut rhalf = 0u32;
            let mut connected = 0.0;
            for &(sqi, pt, am) in &plist[side] {
                if pt != 0 {
                    let free = am & !own;
                    mob[pt - 1] += free.count_ones();
                    smob[pt - 1] += (free & !atk[opp]).count_ones();
                    if pt == 3 {
                        let fmask = file_mask(sqi % 8);
                        if pieces_bb[0] & fmask == 0 {
                            ropen += 1;
                        } else if own_p & fmask == 0 {
                            rhalf += 1;
                        }
                        if am & own_r & !(1u64 << sqi) != 0 {
                            connected = 1.0;
                        }
                    }
                }
                if hung != 0 && (1u64 << sqi) & hung != 0 {
                    hval += TYPE_VAL[pt];
                }
            }
            for ti in 0..4 {
                s += self.w[771 + 4 * side + ti] * (mob[ti] as f64) / 14.0;
                s += self.w[779 + 4 * side + ti] * (smob[ti] as f64) / 14.0;
            }
            s += self.w[789 + side] * hval / 9.0;
            s += self.w[803 + side] * (ropen as f64) / 2.0;
            s += self.w[805 + side] * (rhalf as f64) / 2.0;
            s += self.w[807 + side] * connected;
            // king ring
            let k = pieces_bb[5] & own;
            if k != 0 {
                let ring = bb_u64(get_king_moves(Square::index(k.trailing_zeros() as usize)));
                s += self.w[791 + side] * ((ring & atk[opp]).count_ones() as f64) / 8.0;
            }
            // castling rights (raw)
            let color = if side == 0 { Color::White } else { Color::Black };
            let rights = board.castle_rights(color);
            if rights.short.is_some() {
                s += self.w[793 + 2 * side];
            }
            if rights.long.is_some() {
                s += self.w[794 + 2 * side];
            }
            // pawn structure
            let mut doubled = 0u32;
            let mut isolated = 0u32;
            for f in 0..8 {
                let cnt = (own_p & file_mask(f)).count_ones();
                if cnt > 1 {
                    doubled += cnt - 1;
                }
                if cnt > 0 && own_p & adj_files(f) == 0 {
                    isolated += cnt;
                }
            }
            let mut passed = 0u32;
            let mut m = own_p;
            while m != 0 {
                let sqi = m.trailing_zeros() as usize;
                if front_span(side == 0, sqi) & opp_p == 0 {
                    passed += 1;
                }
                m &= m - 1;
            }
            s += self.w[797 + side] * (doubled as f64) / 8.0;
            s += self.w[799 + side] * (isolated as f64) / 8.0;
            s += self.w[801 + side] * (passed as f64) / 8.0;
        }
        s
    }

    /// side-to-move tanh value (negamax convention)
    fn stm(&self, board: &Board) -> f64 {
        let v = self.score(board).tanh();
        if board.side_to_move() == Color::White {
            v
        } else {
            -v
        }
    }
}

const MATE: f64 = 10.0;

fn moves_of(board: &Board) -> Vec<Move> {
    let mut out = Vec::with_capacity(48);
    board.generate_moves(|ml| {
        out.extend(ml);
        false
    });
    out
}

fn is_capture(board: &Board, mv: Move) -> bool {
    bb_u64(board.colors(!board.side_to_move())) & (1u64 << mv.to as usize) != 0
        || (board.piece_on(mv.from) == Some(Piece::Pawn) && Some(mv.to) == board.en_passant().map(|f| {
            let r = if board.side_to_move() == Color::White { Rank::Sixth } else { Rank::Third };
            Square::new(f, r)
        }))
}

fn victim_value(board: &Board, mv: Move) -> f64 {
    board.piece_on(mv.to).map(|p| TYPE_VAL[p as usize]).unwrap_or(1.0)
}

/// standard-UCI text for a move (cozy uses king-takes-rook for castling)
fn uci(board: &Board, mv: Move) -> String {
    if board.piece_on(mv.from) == Some(Piece::King)
        && bb_u64(board.colors(board.side_to_move())) & (1u64 << mv.to as usize) != 0
    {
        let rank = mv.from.rank();
        let dst = if mv.to.file() > mv.from.file() { File::G } else { File::C };
        return format!("{}{}", mv.from, Square::new(dst, rank));
    }
    format!("{}", mv)
}

// --- v2 (spec :Depth-per-second:): transposition table, killers, history, null move -------
const TT_BITS: usize = 20;                            // 2^20 entries
const TT_SIZE: usize = 1 << TT_BITS;
const BOUND_EXACT: u8 = 0;
const BOUND_LOWER: u8 = 1;
const BOUND_UPPER: u8 = 2;
const MAX_PLY: usize = 64;

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    depth: i32,
    score: f64,
    bound: u8,
    mv: Option<Move>,
}

impl Default for TtEntry {
    fn default() -> Self {
        TtEntry { key: 0, depth: -1, score: 0.0, bound: BOUND_EXACT, mv: None }
    }
}

struct SearchCtx<'a> {
    eval: &'a Eval,
    nodes: u64,
    tt: &'a mut Vec<TtEntry>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: Vec<u64>,                                // from*64+to
}

impl<'a> SearchCtx<'a> {
    fn qsearch(&mut self, board: &Board, mut alpha: f64, beta: f64) -> (f64, Board) {
        self.nodes += 1;
        let moves = moves_of(board);
        if moves.is_empty() {
            let sc = if board.checkers() != BitBoard::EMPTY { -MATE } else { 0.0 };
            return (sc, board.clone());
        }
        let stand = self.eval.stm(board);
        if stand >= beta {
            return (stand, board.clone());
        }
        let mut best = stand;
        let mut best_leaf = board.clone();
        if stand > alpha {
            alpha = stand;
        }
        let mut caps: Vec<Move> = moves.into_iter().filter(|&m| is_capture(board, m)).collect();
        caps.sort_by(|&a, &b| victim_value(board, b).partial_cmp(&victim_value(board, a)).unwrap());
        for mv in caps {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let (sc, leaf) = self.qsearch(&child, -beta, -alpha);
            let sc = -sc;
            if sc > best {
                best = sc;
                best_leaf = leaf;
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                break;
            }
        }
        (best, best_leaf)
    }

    /// returns (score_stm, best_move, leaf_board). v2: TT (cutoffs at NON-PV nodes only, so
    /// the PV leaf stays exact for training), TT-move/killer/history ordering, null-move R=2.
    fn negamax(
        &mut self,
        board: &Board,
        depth: u32,
        ply: usize,
        mut alpha: f64,
        beta: f64,
    ) -> (f64, Option<Move>, Board) {
        self.nodes += 1;
        let moves = moves_of(board);
        if moves.is_empty() {
            let sc = if board.checkers() != BitBoard::EMPTY {
                -(MATE + depth as f64)                      // deeper depth remaining = faster mate
            } else {
                0.0
            };
            return (sc, None, board.clone());
        }
        if depth == 0 {
            let (sc, leaf) = self.qsearch(board, alpha, beta);
            return (sc, None, leaf);
        }
        let is_pv = beta - alpha > 1e-6 * 2.0;             // null-window nodes are non-PV
        let key = board.hash();
        let slot = (key as usize) & (self.tt.len() - 1);   // mask by ACTUAL table size (per-game
                                                           // tables in play_games are smaller)
        let mut tt_mv: Option<Move> = None;
        {
            let e = &self.tt[slot];
            if e.key == key && e.depth >= 0 {
                tt_mv = e.mv;
                if !is_pv && e.depth >= depth as i32 && e.score.abs() < MATE - 1.0 {
                    match e.bound {
                        BOUND_EXACT => return (e.score, e.mv, board.clone()),
                        BOUND_LOWER if e.score >= beta => return (e.score, e.mv, board.clone()),
                        BOUND_UPPER if e.score <= alpha => return (e.score, e.mv, board.clone()),
                        _ => {}
                    }
                }
            }
        }
        // null-move pruning: not in check, not a pawn-only endgame, non-PV, enough depth
        let in_check = board.checkers() != BitBoard::EMPTY;
        if !is_pv && !in_check && depth >= 3 {
            let stm = board.side_to_move();
            let heavy = bb_u64(board.colors(stm))
                & !bb_u64(board.pieces(Piece::Pawn))
                & !bb_u64(board.pieces(Piece::King));
            if heavy != 0 {
                if let Some(nb) = board.null_move() {
                    let (sc, _, _) = self.negamax(&nb, depth - 3, ply + 1, -beta, -beta + 1e-6);
                    if -sc >= beta {
                        return (beta, None, board.clone());
                    }
                }
            }
        }
        // ordering: TT move, captures (MVV-LVA), killers, history-scored quiets
        let mut scored: Vec<(f64, Move)> = moves
            .into_iter()
            .map(|m| {
                let s = if Some(m) == tt_mv {
                    1e12
                } else if is_capture(board, m) {
                    1e9 + victim_value(board, m) * 100.0
                        - board.piece_on(m.from).map(|p| TYPE_VAL[p as usize]).unwrap_or(0.0)
                } else if ply < MAX_PLY && self.killers[ply].contains(&Some(m)) {
                    1e6
                } else {
                    self.history[(m.from as usize) * 64 + (m.to as usize)] as f64
                };
                (s, m)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let alpha0 = alpha;
        let mut best = f64::NEG_INFINITY;
        let mut best_mv = None;
        let mut best_leaf = board.clone();
        let mut idx = 0usize;
        for (_, mv) in scored {
            let mut child = board.clone();
            child.play_unchecked(mv);
            // v3.5 LMR (operator directive: depth via SELECTIVITY, not enumeration): late
            // quiet moves search reduced with a null window; a fail-high triggers a full
            // re-search — surprise gets rigor, boredom gets pruned, the local min stays sound.
            let quiet = !is_capture(board, mv);
            let (sc, leaf) = if idx >= 4 && quiet && depth >= 3 && !in_check {
                let r: u32 = if idx >= 12 { 2 } else { 1 };
                let (s1, _, l1) =
                    self.negamax(&child, depth - 1 - r, ply + 1, -alpha - 1e-6, -alpha);
                if -s1 > alpha {
                    let (s2, _, l2) = self.negamax(&child, depth - 1, ply + 1, -beta, -alpha);
                    (-s2, l2)
                } else {
                    (-s1, l1)
                }
            } else {
                let (s2, _, l2) = self.negamax(&child, depth - 1, ply + 1, -beta, -alpha);
                (-s2, l2)
            };
            idx += 1;
            if sc > best {
                best = sc;
                best_mv = Some(mv);
                best_leaf = leaf;
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                if !is_capture(board, mv) && ply < MAX_PLY {
                    if self.killers[ply][0] != Some(mv) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(mv);
                    }
                    self.history[(mv.from as usize) * 64 + (mv.to as usize)] +=
                        (depth as u64) * (depth as u64);
                }
                break;
            }
        }
        if best.abs() < MATE - 1.0 {
            let bound = if best <= alpha0 {
                BOUND_UPPER
            } else if best >= beta {
                BOUND_LOWER
            } else {
                BOUND_EXACT
            };
            self.tt[slot] = TtEntry { key, depth: depth as i32, score: best, bound, mv: best_mv };
        }
        (best, best_mv, best_leaf)
    }
}

#[pyclass]
struct Searcher {
    eval: Eval,
    tt: Vec<TtEntry>,                                  // persists across calls (reuse within a game)
}

#[pymethods]
impl Searcher {
    #[new]
    fn new(weights: Vec<f64>, bias: f64) -> PyResult<Self> {
        if weights.len() != NFEAT && weights.len() != NAMAP {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} (kc) or {} (amap) weights, got {}",
                NFEAT,
                NAMAP,
                weights.len()
            )));
        }
        Ok(Searcher { eval: Eval { w: weights, bias }, tt: vec![TtEntry::default(); TT_SIZE] })
    }

    /// White-absolute pre-tanh score of a FEN (parity testing)
    fn score(&self, fen: &str) -> PyResult<f64> {
        let board: Board = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e:?}")))?;
        Ok(self.eval.score(&board))
    }

    /// (best_uci, white_value_tanh, leaf_fen, predicted_reply_uci, nodes)
    fn search(&mut self, fen: &str, depth: u32) -> PyResult<(String, f64, String, String, u64)> {
        let board: Board = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e:?}")))?;
        let mut ctx = SearchCtx {
            eval: &self.eval,
            nodes: 0,
            tt: &mut self.tt,
            killers: [[None; 2]; MAX_PLY],
            history: vec![0u64; 64 * 64],
        };
        let mut result = None;
        let mut prev = 0.0f64;
        for d in 1..=depth {
            // v3.5 aspiration windows: search a narrow window around the previous iterate;
            // widen fully on a fail — cheap when the eval is stable, safe when it is not.
            let (mut lo, mut hi) = if d >= 3 {
                (prev - 0.08, prev + 0.08)
            } else {
                (f64::NEG_INFINITY, f64::INFINITY)
            };
            loop {
                let r = ctx.negamax(&board, d, 0, lo, hi);
                if r.0 <= lo && lo.is_finite() {
                    lo = f64::NEG_INFINITY;
                    continue;
                }
                if r.0 >= hi && hi.is_finite() {
                    hi = f64::INFINITY;
                    continue;
                }
                prev = r.0;
                result = Some(r);
                break;
            }
        }
        let (sc, mv, leaf) = result.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("depth 0"))?;
        let mv = mv.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no legal moves"))?;
        // predicted reply: the child's best move at depth-1
        let mut child = board.clone();
        child.play_unchecked(mv);
        let pred = if depth >= 2 {
            let (_, pm, _) = ctx.negamax(&child, depth - 1, 0, f64::NEG_INFINITY, f64::INFINITY);
            pm.map(|m| uci(&child, m)).unwrap_or_default()
        } else {
            String::new()
        };
        let white_val = if board.side_to_move() == Color::White { sc } else { -sc };
        Ok((
            uci(&board, mv),
            white_val.clamp(-1.0, 1.0),
            format!("{leaf}"),
            pred,
            ctx.nodes,
        ))
    }
}

// --- Merge 9 (spec/parallel-generation.spec.md): native parallel game generation ----------
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// one generation game; returns (z_white, agent_white, per-agent-move records)
/// record = (leaf_fen, white_value, predicted: was the reply to the PREVIOUS agent move the
/// one its search expected). opp_depth 0 = uniform-random opponent.
#[allow(clippy::too_many_arguments)]
fn play_one(
    agent: &Eval,
    opp: &Eval,
    opp_depth: u32,
    depth: u32,
    eps: f64,
    tau: f64,
    ply_cap: u32,
    seed: u64,
    agent_white: bool,
) -> (f64, bool, Vec<(String, f64, bool)>) {
    let mut rng = XorShift(seed | 1);
    let mut tt_a = vec![TtEntry::default(); 1 << 16];
    let mut tt_o = vec![TtEntry::default(); 1 << 16];
    let mut board = Board::default();
    let mut recs: Vec<(String, f64, bool)> = Vec::with_capacity(64);
    let mut pending_pred: Option<Move> = None;
    let mut reply_actual: Option<Move> = None;
    let mut plies = 0u32;
    let z;
    loop {
        let moves = moves_of(&board);
        if moves.is_empty() {
            z = if board.checkers() != BitBoard::EMPTY {
                if board.side_to_move() == Color::White { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            break;
        }
        if plies >= ply_cap {
            z = 0.0;
            break;
        }
        let my_turn = (board.side_to_move() == Color::White) == agent_white;
        let mv;
        if my_turn {
            if eps > 0.0 && rng.unit() < eps {
                mv = moves[rng.below(moves.len())];         // ε: state-coverage exploration
                let predicted = pending_pred.is_some() && reply_actual == pending_pred;
                let _ = predicted;                          // ε-moves record nothing (no search value)
                pending_pred = None;
                reply_actual = None;
            } else {
                let mut ctx = SearchCtx {
                    eval: agent,
                    nodes: 0,
                    tt: &mut tt_a,
                    killers: [[None; 2]; MAX_PLY],
                    history: vec![0u64; 64 * 64],
                };
                let (sc, bm, leaf) = ctx.negamax(&board, depth, 0, f64::NEG_INFINITY, f64::INFINITY);
                let best = bm.unwrap_or(moves[0]);
                // :Softmax-tau: (Merge 11): optimism-directed exploration — sample the move
                // ∝ exp(σ·v1/τ) over 1-ply afterstate values. Inflated (optimistic-init)
                // values attract visits until data deflates them toward truth; τ scales how
                // strongly value differences steer. A non-best sample records nothing (the
                // search value describes best play, not the sampled move — same contract as
                // ε-moves). tau=0 = greedy (unchanged behavior).
                let mut chosen = best;
                if tau > 0.0 {
                    let sign = if board.side_to_move() == Color::White { 1.0 } else { -1.0 };
                    let vals: Vec<f64> = moves
                        .iter()
                        .map(|&m| {
                            let mut c = board.clone();
                            c.play_unchecked(m);
                            sign * agent.score(&c).tanh()
                        })
                        .collect();
                    let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let ws: Vec<f64> = vals.iter().map(|v| ((v - mx) / tau).exp()).collect();
                    let tot: f64 = ws.iter().sum();
                    let mut r = rng.unit() * tot;
                    for (i, wgt) in ws.iter().enumerate() {
                        r -= wgt;
                        if r <= 0.0 {
                            chosen = moves[i];
                            break;
                        }
                    }
                }
                let predicted = pending_pred.is_some() && reply_actual == pending_pred;
                if chosen == best {
                    let white_val = if board.side_to_move() == Color::White { sc } else { -sc };
                    recs.push((format!("{leaf}"), white_val.clamp(-1.0, 1.0), predicted));
                    // predicted reply for the NEXT record: child's best at depth-1
                    let mut child = board.clone();
                    child.play_unchecked(best);
                    pending_pred = if depth >= 2 {
                        let (_, pm, _) = ctx.negamax(&child, depth - 1, 0, f64::NEG_INFINITY, f64::INFINITY);
                        pm
                    } else {
                        None
                    };
                } else {
                    // sampled off-best move: record ITS afterstate + 1-ply value (the python
                    // softmax path's contract for unexpanded roots) — on-policy data, no
                    // prediction to carry forward.
                    let mut child = board.clone();
                    child.play_unchecked(chosen);
                    let v_white = agent.score(&child).tanh();
                    recs.push((format!("{child}"), v_white.clamp(-1.0, 1.0), predicted));
                    pending_pred = None;
                }
                reply_actual = None;
                mv = chosen;
            }
        } else {
            if opp_depth == 0 {
                mv = moves[rng.below(moves.len())];
            } else {
                let mut ctx = SearchCtx {
                    eval: opp,
                    nodes: 0,
                    tt: &mut tt_o,
                    killers: [[None; 2]; MAX_PLY],
                    history: vec![0u64; 64 * 64],
                };
                let (_, bm, _) = ctx.negamax(&board, opp_depth, 0, f64::NEG_INFINITY, f64::INFINITY);
                mv = bm.unwrap_or(moves[0]);
            }
            if reply_actual.is_none() {
                reply_actual = Some(mv);
            }
        }
        board.play_unchecked(mv);
        plies += 1;
    }
    (z, agent_white, recs)
}

/// Parallel generation: n_games across `threads` OS threads. Opponent = (opp_weights, opp_depth);
/// opp_depth 0 ignores opp_weights (uniform random). Returns [(z, agent_white,
/// [(leaf_fen, white_value, predicted)])]. seed varies per game; colors alternate.
#[pyfunction]
#[pyo3(signature = (weights, bias, opp_weights, opp_bias, opp_depth, n_games, threads, depth, eps, ply_cap, seed, tau=0.0))]
#[allow(clippy::too_many_arguments)]
fn play_games(
    py: Python<'_>,
    weights: Vec<f64>,
    bias: f64,
    opp_weights: Vec<f64>,
    opp_bias: f64,
    opp_depth: u32,
    n_games: usize,
    threads: usize,
    depth: u32,
    eps: f64,
    ply_cap: u32,
    seed: u64,
    tau: f64,
) -> PyResult<Vec<(f64, bool, Vec<(String, f64, bool)>)>> {
    for wl in [weights.len(), opp_weights.len()] {
        if wl != NFEAT && wl != NAMAP {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "weights must be 809-dim (kc) or 897-dim (amap)"));
        }
    }
    let agent = Eval { w: weights, bias };
    let opp = Eval { w: opp_weights, bias: opp_bias };
    let results = py.allow_threads(|| {
        let agent = &agent;
        let opp = &opp;
        let nt = threads.max(1);
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..nt {
                handles.push(s.spawn(move || {
                    let mut out = Vec::new();
                    let mut g = t;
                    while g < n_games {
                        out.push((
                            g,
                            play_one(agent, opp, opp_depth, depth, eps, tau, ply_cap,
                                     seed.wrapping_add(g as u64).wrapping_mul(0x9E3779B97F4A7C15),
                                     g % 2 == 0),
                        ));
                        g += nt;
                    }
                    out
                }));
            }
            let mut all: Vec<(usize, (f64, bool, Vec<(String, f64, bool)>))> =
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
            all.sort_by_key(|x| x.0);
            all.into_iter().map(|x| x.1).collect::<Vec<_>>()
        })
    });
    Ok(results)
}

#[pyfunction]
fn version() -> &'static str {
    "v3.7-amap"
}

// --- :Hyb-native: (spec/search-arms.spec.md :Lambda-tree-backup: + :AB-leaf:) -------------
// Native port of chessdq/puct_value.py's puct_move/puct_choose, sel=puct only (hyb's arm
// space never uses ucbv — see search-arms.spec.md). Reuses the SAME Eval (amap-897 linear,
// already native) for expansion values and the SAME negamax for :AB-leaf: depth-d backed
// leaves, so this is a tree/selection layer over infrastructure that already existed.
//
// Known v1 simplification vs the Python engine (disposed explicitly, not silent): move
// choice at tau<=0 is plain visit-argmax, WITHOUT the Python engine's repetition-refusal
// walk (cozy_chess::Board carries no built-in repetition-history API at this call site).
// Acceptable for a screen-tier native port; would need a caller-supplied position-hash
// history to restore bit-for-bit if this graduates past screening.
struct HNode {
    board: Board,
    parent: Option<usize>,
    mv: Option<Move>,
    children: Vec<usize>,
    p: f64,
    n: u32,
    w: f64,
    m: f64,           // minimax subtree value, White-absolute
    v0: f64,           // immutable expansion-batch value (mirrors Python's V0)
    expanded: bool,
    terminal_v: Option<f64>,
}

fn h_terminal_value(board: &Board) -> f64 {
    if board.checkers() != BitBoard::EMPTY {
        return if board.side_to_move() == Color::White { -1.0 } else { 1.0 };
    }
    0.0
}

#[pyclass]
struct HybSearcher {
    eval: Eval,
    tt: Vec<TtEntry>,
    rng: XorShift,
}

impl HybSearcher {
    /// depth-d backed leaf value (:AB-leaf:), White-absolute, using the searcher's OWN
    /// eval + tt (mirrors Searcher::search's white_val conversion).
    fn ab_leaf(&mut self, board: &Board, depth: u32) -> f64 {
        let mut ctx = SearchCtx {
            eval: &self.eval,
            nodes: 0,
            tt: &mut self.tt,
            killers: [[None; 2]; MAX_PLY],
            history: vec![0u64; 64 * 64],
        };
        let (sc, _, _) = ctx.negamax(board, depth.max(1), 0, f64::NEG_INFINITY, f64::INFINITY);
        if board.side_to_move() == Color::White { sc } else { -sc }
    }

    /// Expand `idx`'s children (softmax-over-child-value prior, mover-perspective ranking,
    /// White-absolute storage — exact port of puct_value._expand). Returns the node's
    /// backed leaf value (1-ply minimax by default, or the AB-leaf depth-d value).
    fn expand(&mut self, arena: &mut Vec<HNode>, idx: usize, t_prior: f64, leaf_depth: u32) -> f64 {
        let board = arena[idx].board.clone();
        let moves = moves_of(&board);
        let sgn = if board.side_to_move() == Color::White { 1.0 } else { -1.0 };
        let mut child_idxs = Vec::with_capacity(moves.len());
        let mut zs = Vec::with_capacity(moves.len());
        for mv in &moves {
            let mut cb = board.clone();
            cb.play_unchecked(*mv);
            let terminal_v = if moves_of(&cb).is_empty() { Some(h_terminal_value(&cb)) } else { None };
            let cv = terminal_v.unwrap_or_else(|| self.eval.score(&cb).tanh());
            let z = sgn * cv;
            let cidx = arena.len();
            arena.push(HNode {
                board: cb, parent: Some(idx), mv: Some(*mv), children: Vec::new(),
                p: 0.0, n: 0, w: 0.0, m: cv, v0: cv, expanded: false, terminal_v,
            });
            child_idxs.push(cidx);
            zs.push(z);
        }
        let zmax = zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut psum = 0.0;
        let mut ps = Vec::with_capacity(zs.len());
        for &z in &zs {
            let p = ((z - zmax) / t_prior).exp();
            psum += p;
            ps.push(p);
        }
        for (i, &cidx) in child_idxs.iter().enumerate() {
            arena[cidx].p = ps[i] / psum;
        }
        let mut backed = sgn * zmax;
        if leaf_depth > 0 {
            backed = self.ab_leaf(&board, leaf_depth);
        }
        arena[idx].children = child_idxs;
        arena[idx].expanded = true;
        arena[idx].m = backed;
        backed
    }

    fn minimax_children(arena: &[HNode], idx: usize) -> f64 {
        let white = arena[idx].board.side_to_move() == Color::White;
        let ms = arena[idx].children.iter().map(|&c| arena[c].m);
        if white { ms.fold(f64::NEG_INFINITY, f64::max) } else { ms.fold(f64::INFINITY, f64::min) }
    }
}

#[pymethods]
impl HybSearcher {
    #[new]
    fn new(weights: Vec<f64>, bias: f64, seed: u64) -> PyResult<Self> {
        if weights.len() != NFEAT && weights.len() != NAMAP {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} (kc) or {} (amap) weights, got {}", NFEAT, NAMAP, weights.len())));
        }
        Ok(HybSearcher {
            eval: Eval { w: weights, bias },
            tt: vec![TtEntry::default(); TT_SIZE],
            rng: XorShift(seed | 1),
        })
    }

    /// (best_uci, gv_white_tanh, leaf_fen, predicted_reply_uci, nodes) — drop-in analog of
    /// puct_choose for the TDLeaf 4-tuple contract (spec/search-arms.spec.md
    /// :Lambda-target-training:). leaf_depth=0 -> vf (1-ply backed) leaf; >0 -> :AB-leaf:.
    #[pyo3(signature = (fen, sims, lambda_backup=0.0, c_puct=1.5, t_prior=0.2, leaf_depth=0, tau=0.0))]
    #[allow(clippy::too_many_arguments)]
    fn choose(
        &mut self, fen: &str, sims: u32, lambda_backup: f64, c_puct: f64, t_prior: f64,
        leaf_depth: u32, tau: f64,
    ) -> PyResult<(String, f64, String, String, u64)> {
        let board: Board = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e:?}")))?;
        if moves_of(&board).is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("choose on terminal position"));
        }
        let lb = lambda_backup;
        let mut arena: Vec<HNode> = vec![HNode {
            board: board.clone(), parent: None, mv: None, children: Vec::new(),
            p: 0.0, n: 1, w: 0.0, m: 0.0, v0: 0.0, expanded: false, terminal_v: None,
        }];
        let backed = self.expand(&mut arena, 0, t_prior, leaf_depth);
        arena[0].w = backed;
        let mut nodes: u64 = 1;

        for _ in 0..sims {
            // SELECT
            let mut idx = 0usize;
            while arena[idx].expanded && arena[idx].terminal_v.is_none() {
                let sgn = if arena[idx].board.side_to_move() == Color::White { 1.0 } else { -1.0 };
                let sqrt_n = (arena[idx].n as f64).sqrt();
                let mut best = arena[idx].children[0];
                let mut best_s = f64::NEG_INFINITY;
                for &cidx in &arena[idx].children {
                    let kid = &arena[cidx];
                    let q = if kid.n == 0 { 0.0 } else { kid.w / kid.n as f64 };
                    let q_eff = if lb == 0.0 { q } else { (1.0 - lb) * q + lb * kid.m };
                    let u = c_puct * kid.p * sqrt_n / (1.0 + kid.n as f64);
                    let s = sgn * q_eff + u;
                    if s > best_s { best = cidx; best_s = s; }
                }
                idx = best;
            }
            // EVALUATE
            nodes += 1;
            let v = match arena[idx].terminal_v {
                Some(tv) => tv,
                None => self.expand(&mut arena, idx, t_prior, leaf_depth),
            };
            // BACKUP (leaf keeps its own M until its own children get visited)
            let leaf = idx;
            let mut cur = Some(idx);
            while let Some(node) = cur {
                arena[node].n += 1;
                arena[node].w += v;
                if lb > 0.0 && arena[node].expanded && node != leaf {
                    arena[node].m = Self::minimax_children(&arena, node);
                }
                cur = arena[node].parent;
            }
        }

        // MOVE CHOICE — visit-argmax (tau>0: softmax dither); no repetition-refusal (v1 note above)
        let root_children = arena[0].children.clone();
        let chosen_idx = if tau > 0.0 {
            let sgn = if board.side_to_move() == Color::White { 1.0 } else { -1.0 };
            let vals: Vec<f64> = root_children.iter().map(|&c| {
                let kid = &arena[c];
                let mean = (kid.w + kid.v0) / (kid.n as f64 + 1.0);
                sgn * ((1.0 - lb) * mean + lb * kid.m)
            }).collect();
            let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ws: Vec<f64> = vals.iter().map(|v| ((v - mx) / tau).exp()).collect();
            let total: f64 = ws.iter().sum();
            let r = self.rng.unit() * total;
            let mut acc = 0.0;
            let mut pick = root_children.len() - 1;
            for (i, w) in ws.iter().enumerate() {
                acc += w;
                if r <= acc { pick = i; break; }
            }
            root_children[pick]
        } else {
            *root_children.iter().max_by_key(|&&c| arena[c].n).unwrap()
        };

        // gv: lambda-blend of root mean and EXPANDED-children minimax (:Backed-bootstrap:)
        let backed_ms: Vec<f64> = root_children.iter()
            .filter(|&&c| arena[c].expanded || arena[c].terminal_v.is_some())
            .map(|&c| arena[c].m)
            .collect();
        let root_q = arena[0].w / arena[0].n as f64;
        let gv = if backed_ms.is_empty() {
            root_q
        } else {
            let white = board.side_to_move() == Color::White;
            let mm = if white { backed_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
                     else { backed_ms.iter().cloned().fold(f64::INFINITY, f64::min) };
            (1.0 - lb) * root_q + lb * mm
        };

        // PV leaf: descend the chosen child by max visits while real tree exists
        let mut pv = chosen_idx;
        loop {
            if !arena[pv].expanded || arena[pv].children.is_empty() { break; }
            let best_child = *arena[pv].children.iter().max_by_key(|&&c| arena[c].n).unwrap();
            if arena[best_child].n == 0 { break; }
            pv = best_child;
        }
        let pred = if arena[chosen_idx].expanded && !arena[chosen_idx].children.is_empty() {
            let best_reply = *arena[chosen_idx].children.iter().max_by_key(|&&c| arena[c].n).unwrap();
            if arena[best_reply].n > 0 { uci(&arena[chosen_idx].board, arena[best_reply].mv.unwrap()) }
            else { String::new() }
        } else { String::new() };

        Ok((
            uci(&board, arena[chosen_idx].mv.unwrap()),
            gv.clamp(-1.0, 1.0),
            format!("{}", arena[pv].board),
            pred,
            nodes,
        ))
    }
}

#[pymodule]
fn rsearch4(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Searcher>()?;
    m.add_class::<HybSearcher>()?;
    m.add_function(wrap_pyfunction!(play_games, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
