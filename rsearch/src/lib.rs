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
    w: [f64; NFEAT],
    bias: f64,
}

impl Eval {
    /// White-absolute pre-tanh linear score; feature semantics == cem_loop.encode_features.
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

struct SearchCtx<'a> {
    eval: &'a Eval,
    nodes: u64,
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

    /// returns (score_stm, best_move, leaf_board)
    fn negamax(&mut self, board: &Board, depth: u32, mut alpha: f64, beta: f64) -> (f64, Option<Move>, Board) {
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
        // order: captures by victim value first
        let mut ordered = moves;
        ordered.sort_by(|&a, &b| {
            let ca = if is_capture(board, a) { victim_value(board, a) } else { -1.0 };
            let cb = if is_capture(board, b) { victim_value(board, b) } else { -1.0 };
            cb.partial_cmp(&ca).unwrap()
        });
        let mut best = f64::NEG_INFINITY;
        let mut best_mv = None;
        let mut best_leaf = board.clone();
        for mv in ordered {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let (sc, _, leaf) = self.negamax(&child, depth - 1, -beta, -alpha);
            let sc = -sc;
            if sc > best {
                best = sc;
                best_mv = Some(mv);
                best_leaf = leaf;
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                break;
            }
        }
        (best, best_mv, best_leaf)
    }
}

#[pyclass]
struct Searcher {
    eval: Eval,
}

#[pymethods]
impl Searcher {
    #[new]
    fn new(weights: Vec<f64>, bias: f64) -> PyResult<Self> {
        if weights.len() != NFEAT {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} weights, got {}",
                NFEAT,
                weights.len()
            )));
        }
        let mut w = [0.0; NFEAT];
        w.copy_from_slice(&weights);
        Ok(Searcher { eval: Eval { w, bias } })
    }

    /// White-absolute pre-tanh score of a FEN (parity testing)
    fn score(&self, fen: &str) -> PyResult<f64> {
        let board: Board = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e:?}")))?;
        Ok(self.eval.score(&board))
    }

    /// (best_uci, white_value_tanh, leaf_fen, predicted_reply_uci, nodes)
    fn search(&self, fen: &str, depth: u32) -> PyResult<(String, f64, String, String, u64)> {
        let board: Board = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e:?}")))?;
        let mut ctx = SearchCtx { eval: &self.eval, nodes: 0 };
        let mut result = None;
        for d in 1..=depth {
            result = Some(ctx.negamax(&board, d, f64::NEG_INFINITY, f64::INFINITY));
        }
        let (sc, mv, leaf) = result.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("depth 0"))?;
        let mv = mv.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no legal moves"))?;
        // predicted reply: the child's best move at depth-1
        let mut child = board.clone();
        child.play_unchecked(mv);
        let pred = if depth >= 2 {
            let (_, pm, _) = ctx.negamax(&child, depth - 1, f64::NEG_INFINITY, f64::INFINITY);
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

#[pymodule]
fn rsearch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Searcher>()?;
    Ok(())
}
